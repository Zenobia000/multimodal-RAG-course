# AI Evolution: Paper Collection

This directory contains 31 papers from "AI演義：36篇論文開啟你的探索之旅".

## 📊 Download Statistics

- **Total papers downloaded**: 31 PDFs ✓
- **Course items**: 33 (includes 1 blog post + 1 paywalled paper)
- **Successfully downloaded**: 31
- **Requires institutional access**: 1 (Brook for GPUs - ACM)
- **Not applicable**: 1 (The Bitter Lesson - blog post only)

## 📁 Directory Structure

```
papers/
├── 01_model_paradigm/     # 模型的范式變遷 (10 papers)
├── 02_infrastructure/      # Infra與資料的變遷 (5 papers)
├── 03_language_models/     # 語言模型的發展 (8 papers)
└── 04_multimodal/          # 多模態模型的發展 (8 papers)

Total: 31 PDFs (~170 MB)
```

## 📚 Papers by Category

### 01_model_paradigm - 模型的范式變遷 (10 papers)

| Year | Paper | File | Status |
|------|-------|------|--------|
| 2004 | Brook for GPUs | - | ⚠️ Requires ACM access |
| 2012 | AlexNet | 2012_AlexNet.pdf | ✓ |
| 2014 | Sequence to Sequence Learning | 2014_Seq2Seq.pdf | ✓ |
| 2015 | Knowledge Distillation | 2015_Knowledge_Distillation.pdf | ✓ |
| 2015 | ResNet | 2015_ResNet.pdf | ✓ |
| 2017 | Attention Is All You Need (Transformer) | 2017_Transformer.pdf | ✓ |
| 2017 | AlphaGo Zero | 2017_AlphaGo_Zero.pdf | ✓ |
| 2017 | Mixture-of-Experts (MoE) | 2017_MoE.pdf | ✓ |
| 2021 | LoRA | 2021_LoRA.pdf | ✓ |
| 2022 | Chain-of-Thought | 2022_Chain_of_Thought.pdf | ✓ |
| 2022 | ReAct | 2022_ReAct.pdf | ✓ |

### 02_infrastructure - Infra與資料的變遷 (5 papers)

| Year | Paper | File | Status |
|------|-------|------|--------|
| 2018 | The Bitter Lesson | - | ℹ️ Blog post only |
| 2019 | ZeRO | 2019_ZeRO.pdf | ✓ |
| 2020 | Scaling Laws | 2020_Scaling_Laws.pdf | ✓ |
| 2022 | LAION-5B | 2022_LAION_5B.pdf | ✓ |
| 2023 | RefinedWeb | 2023_RefinedWeb.pdf | ✓ |
| 2024 | MegaScale | 2024_MegaScale.pdf | ✓ |

### 03_language_models - 語言模型的發展 (8 papers)

| Year | Paper | File | Status |
|------|-------|------|--------|
| 2013 | Word2Vec | 2013_Word2Vec.pdf | ✓ |
| 2016 | Google NMT | 2016_Google_NMT.pdf | ✓ |
| 2018 | GPT-1 | 2018_GPT_1.pdf | ✓ |
| 2018 | BERT | 2018_BERT.pdf | ✓ |
| 2019 | GPT-2 | 2019_GPT_2.pdf | ✓ |
| 2020 | GPT-3 | 2020_GPT_3.pdf | ✓ |
| 2022 | InstructGPT | 2022_InstructGPT.pdf | ✓ |
| 2024 | TULU 3 | 2024_TULU_3.pdf | ✓ |

### 04_multimodal - 多模態模型的發展 (8 papers)

| Year | Paper | File | Status |
|------|-------|------|--------|
| 2014 | DeepVideo | 2014_DeepVideo.pdf | ✓ |
| 2014 | Two-Stream CNN | 2014_Two_Stream_CNN.pdf | ✓ |
| 2015 | GAN | 2015_GAN.pdf | ✓ |
| 2020 | DDPM | 2020_DDPM.pdf | ✓ |
| 2020 | Vision Transformer (ViT) | 2020_ViT.pdf | ✓ |
| 2021 | CLIP | 2021_CLIP.pdf | ✓ |
| 2022 | Stable Diffusion | 2022_Stable_Diffusion.pdf | ✓ |
| 2022 | DiT | 2022_DiT.pdf | ✓ |

## 🔗 Manual Download Required

### Brook for GPUs (2004)
- **URL**: https://dl.acm.org/doi/pdf/10.1145/1015706.1015800
- **Reason**: ACM Digital Library requires institutional subscription
- **Alternative**: Search on Google Scholar or ResearchGate

### The Bitter Lesson (2018)
- **URL**: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- **Note**: Blog post by Richard Sutton, not a formal paper

## 🔄 Re-running the Download

To update or re-download papers:

```bash
cd d:\python_workspace\project_nlp\data_governance
python download_papers.py
```

The script will:
- ✓ Skip already downloaded files
- ✓ Auto-create missing directories
- ✓ Show progress for each download
- ✓ Generate a detailed summary report
- ✓ Rate limit (1 second between requests)

## 📖 File Naming Convention

All files follow: `YYYY_PaperName.pdf`

Examples:
- `2017_Transformer.pdf`
- `2018_BERT.pdf`
- `2022_Stable_Diffusion.pdf`

## 📦 Collection Details

- **File formats**: PDF
- **Total size**: ~170 MB
- **Year range**: 2004-2024
- **Sources**: arXiv (26), OpenAI (2), NeurIPS/CVPR (2), Nature (1)
- **Download time**: ~2-3 minutes (depending on network)

## 🌟 Key Papers by Impact

**Foundational (>100k citations)**:
- AlexNet (2012) - 145k citations
- BERT (2018) - 130k citations
- ResNet (2015) - 200k citations
- Transformer (2017) - 320k citations

**Essential (>50k citations)**:
- Word2Vec (2013) - 95k citations
- Seq2Seq (2014) - 70k citations
- GAN (2015) - 60k citations
- ViT (2020) - 55k citations
- CLIP (2021) - 40k citations
- GPT-2 (2019) - 40k citations

## 📝 Notes

1. All papers organized by primary research contribution
2. arXiv papers downloaded directly from arxiv.org
3. Some papers have multiple versions - latest version downloaded
4. Papers stored in categorized subdirectories
5. Download script includes automatic retry on failure

## 🔧 Troubleshooting

**If download fails:**
1. Check internet connection
2. Verify disk space (need ~200 MB)
3. Some servers may temporarily block automated downloads
4. Try again after a few minutes

**If file is corrupted:**
Delete the PDF and re-run the script to re-download.

## 📚 Additional Resources

- **arXiv**: https://arxiv.org/
- **Papers with Code**: https://paperswithcode.com/
- **Google Scholar**: https://scholar.google.com/
- **Semantic Scholar**: https://www.semanticscholar.org/

---

**Last Updated**: 2025-01-17
**Script**: `download_papers.py`
**Total Papers**: 31/33 available items
