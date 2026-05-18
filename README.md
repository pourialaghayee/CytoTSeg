# CytoTSeg

CytoTSeg is a morphology-aware teacher-student segmentation framework for bright-field and imaging flow cytometry. The project combines offline benchmarking of candidate teacher models with knowledge distillation into lightweight student networks, enabling efficient cell segmentation while preserving morphology features relevant for downstream analysis.

The framework is designed for settings in which segmentation serves not only as a pixel-level prediction task, but also as a basis for extracting biologically meaningful cell descriptors. To support this goal, CytoTSeg evaluates both segmentation quality and agreement of morphology features with expert-annotated ground truth, with particular emphasis on area and deformation across datasets.
