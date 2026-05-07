import os
import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import BaseLoop
from mmpretrain.registry import LOOPS
from PIL import Image as PillowImage


@LOOPS.register_module()
class ExtractLoop(BaseLoop):
    """Loop for feature extraction.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
    """
    def __init__(self, runner, dataloader, save_path="./data/avec/extracted_features"):
        super().__init__(runner, dataloader)
        self.save_path = save_path

    def run(self):
        """Launch feature extraction."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            data_batch_sequence = data_batch
            data_batch_sequence = self.process_sequence(data_batch_sequence)
            subj_frame_segmet_feat = []
            for data_batch in data_batch_sequence["tensors"]:
                with torch.no_grad():
                    self.runner.call_hook(
                        'before_test_iter', batch_idx=idx, data_batch=data_batch)

                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor
                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model.extract_feat(data_batch["inputs"])
                    subj_frame_segmet_feat.append(feats)
            subj_frame_segmet_feat = torch.cat(subj_frame_segmet_feat, dim=0).cpu().numpy()
            self.save_feat(data_batch_sequence, subj_frame_segmet_feat)

    def process_sequence(self, data_seq):
        """Preprocess data batch.

        Args:
            data_batch (dict): A batch of data.

        Returns:
            dict: A batch of data after preprocessing.
        """
        seq_aug = self.dataloader.dataset.seq_aug
        img_segments = []
        for _, img_seg in data_seq["segments"].items():
            if seq_aug:
                imgs_list = {
                    "img": [PillowImage.open(img[0]) for img in img_seg],
                    "gt_label": data_seq['gt_label'],
                    "subj_id": data_seq['subj_id'],
                }
                imgs_seq = self.dataloader.dataset.pipeline(imgs_list)   

                imgs_ten = torch.stack(imgs_seq["inputs"])
            else:
                processed_img_seg = []
                for img in img_seg:
                    img = dict(
                        img_path=img[0],
                        gt_label=data_seq['gt_label'][0],
                        subj_id=data_seq['subj_id'][0]
                    )
                    img = self.dataloader.dataset.pipeline(img)
                    processed_img_seg.append(img)
                imgs_ten = torch.stack([img["inputs"] for img in processed_img_seg])
            img_segments.append({"inputs": imgs_ten[None]})  # add None to add a batch dimension
        data_seq['tensors'] = img_segments
        return data_seq

    def save_feat(self, data_batch_sequence, subj_frame_segmet_feat):
        """Save features.

        Args:
            data_batch_sequence (dict): A batch of data.
            subj_frame_segmet_feat (list): A list of features.
        """
        split = data_batch_sequence["subj_id"][0].split("_")[-1]
        data_root = self.dataloader.dataset._data_root
        save_dir = os.path.join(data_root, "extracted_features", split)
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{data_batch_sequence['subj_id'][0]}.pkl"
        save_path = os.path.join(save_dir, file_name)

        save_data = {
            "subj_id": data_batch_sequence["subj_id"][0],
            "gt_label": data_batch_sequence["gt_label"][0],
            "segment": data_batch_sequence["segments"],
            "features": subj_frame_segmet_feat
        }
        with open(save_path, "wb") as f:
            torch.save(save_data, f)
        print(f"Save features to {save_path}")
