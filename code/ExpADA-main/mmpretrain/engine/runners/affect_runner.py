import copy
from mmengine.runner import Runner
from mmengine.runner import BaseLoop
from mmpretrain.registry import LOOPS
from mmpretrain.engine.runners.extract_loop import ExtractLoop


class AffectRunner(Runner):
    """Runner for feature extraction.

    Args:
        cfg (Config): Configuration object.
    """
    def __init__(
            self, *,
            extract_dataloader=None,
            extract_cfg=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # default extract dataloader
        if extract_dataloader is None:
            extract_dataloader = self.cfg.get('extract_dataloader', None)
            # extract_dataloader = dict(
            #     dataset=dict(
            #         type="ExtractAVECDataset",
            #         data_root="data/avec_2014_northwind",
            #         split=["Training", "Development", "Testing"],
            #         label_file="label_dict.json",
            #         seg_len=32,
            #         steps=32,
            #         img_type="jpg",
            #         pipeline=[
            #             dict(type='LoadImageFromFile'),
            #             dict(type='ResizeEdge', scale=224, edge='short'),
            #             dict(type='PackInputs', algorithm_keys=["subj_id",]),
            #         ]
            #     ),
            #     sampler=dict(type="DefaultSampler", shuffle=False),
            #     batch_size=1,
            #     num_workers=0,
            # )
        self._extract_dataloader = extract_dataloader
        # default extract loop
        if extract_cfg is None:
            extract_cfg = {}  # type: ignore
        self._extract_loop = extract_cfg    # type: ignore

    @property
    def extract_loop(self):
        """:obj:`BaseLoop`: A loop to run testing."""
        if isinstance(self._extract_loop, BaseLoop) or self._extract_loop is None:
            return self._extract_loop
        else:
            self._extract_loop = self.build_extract_loop(self._extract_loop)
            return self._extract_loop

    def build_extract_loop(self, loop):
        """Build extract loop.

        Examples of ``loop``:

            # `ExtractLoop` will be used
            loop = dict()

            # custom validation loop
            loop = dict(type='CustomValLoop')

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'extract_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._extract_dataloader,
                )
            )
        else:
            loop = ExtractLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._extract_dataloader,
            )  # type: ignore

        return loop  # type: ignore

    def extract(self):
        """Extract features."""

        if self._extract_loop is None:
            raise RuntimeError(
                '`self._extract_loop` should not be None when calling extract '
                'method. Please provide `extract_dataloader`, `extract_cfg` and '
            )

        self._extract_loop = self.build_extract_loop(self._extract_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        self.extract_loop.run()  # type: ignore
        self.call_hook('after_run')
