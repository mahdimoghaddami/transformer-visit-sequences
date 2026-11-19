"""Feature schema and diagnosis coding for TADPOLE visits."""

COGNITIVE = (
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "MMSE",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "MOCA",
    "FAQ",
)
MRI = ("Entorhinal", "Fusiform", "Hippocampus", "ICV", "MidTemp", "Ventricles", "WholeBrain")
BIOMARKERS = (
    "AV45",
    "FDG",
    "ABETA_UPENNBIOMK9_04_19_17",
    "TAU_UPENNBIOMK9_04_19_17",
    "PTAU_UPENNBIOMK9_04_19_17",
)
MEASURES = COGNITIVE + MRI + BIOMARKERS
CATEGORIES = {"cognitive": COGNITIVE, "mri": MRI, "biomarkers": BIOMARKERS}
DXCHANGE = {1: 0, 7: 0, 9: 0, 2: 1, 4: 1, 8: 1, 3: 2, 5: 2, 6: 2}
CLASSES = ("CN", "MCI", "AD")


def select_features(include_time=False, remove=None, only=None):
    if remove and only:
        raise ValueError("Choose either remove or only.")
    groups = dict(CATEGORIES, biomarkers=BIOMARKERS[2:])
    names = list(COGNITIVE + MRI + BIOMARKERS[2:]) if remove or only else list(MEASURES)
    if remove:
        names = [f for f in names if f not in groups[remove]]
    if only:
        names = list(groups[only])
    if include_time:
        names.append("months_to_pred")
    return names + ["dx_num"]
