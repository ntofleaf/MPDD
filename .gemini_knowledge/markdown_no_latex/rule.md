# 规则：Markdown Artifact 中禁止使用 LaTeX 数学公式

## 问题描述

在 Markdown artifact 中使用 LaTeX 数学公式（如 `$...$`、`\hat{}`、`\mathbb{R}`、`\odot`、`\oplus`、`\otimes`、`\rho`、`\phi`、`\eta` 等）会导致渲染时显示为乱码，用户无法阅读。

## 规则

**在所有 Markdown artifact 中，绝对不要使用 LaTeX 数学符号或公式。** 必须使用以下替代方案：

### 替代方案对照表

| LaTeX 语法 | 替代写法 | 示例 |
|:---|:---|:---|
| `$X_1$` | X₁（Unicode 下标） | X₁, X₂, X₃ |
| `$X^n$` | Xⁿ（Unicode 上标） | Hⁿ, Wⁿ |
| `$\hat{Y}$` | Ŷ 或 Y_hat | Ŷ(A→B) |
| `$\mathbb{R}$` | ℝ 或直接写 "实数空间" | ℝ |
| `$\odot$` | ⊙ | A ⊙ B |
| `$\oplus$` | ⊕ | A ⊕ B |
| `$\otimes$` | ⊗ | A ⊗ B |
| `$\rho$` | ρ | ρ |
| `$\phi$` | φ | φ |
| `$\eta$` | η | η |
| `$\alpha$` | α | α |
| `$W_{gate}$` | W_gate | W_gate |
| `$C' \times H^n \times W^n$` | C' × Hⁿ × Wⁿ | 维度为 C' × Hⁿ × Wⁿ |
| `$p\%$` | p% | top p% |
| `$\in$` | ∈ | x ∈ ℝ |
| `$\leq$` | ≤ | a ≤ b |
| `$\geq$` | ≥ | a ≥ b |
| `$\rightarrow$` | → | A → B |
| `$\sum$` | Σ 或 "求和" | Σ(xᵢ) |

### 总结原则

1. **用 Unicode 数学符号**代替 LaTeX 命令（⊙ ⊕ ⊗ ∈ → ≤ ≥ α β γ 等）
2. **用 Unicode 上下标**代替 LaTeX 上下标（₁₂₃ⁿᵢ 等）
3. **用纯文本描述**代替复杂公式
4. **公式放在代码块内**用纯文本格式书写（如 `Y = X ⊙ w + b`）

### 常用 Unicode 上下标参考

- 下标数字: ₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉
- 下标字母: ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
- 上标数字: ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹
- 上标字母: ⁿ ⁱ
