"""The competition's study ID and output order."""

STUDY_ID = "StudyInstanceUID"
SERIES_ID = "SeriesInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES = ["Sagittal", "Coronal", "Axial"]
# Per-window descriptors for the Raptor recipe: plane, sequence flags, knee side
# (+1 left, -1 right, 0 unknown), and the anatomical offset of the middle slice.
WINDOW_FEATURES = ["sagittal", "coronal", "axial", "fluid_sensitive", "fat_suppressed", "left_knee", "offset_40mm"]
SIDE_FEATURE = WINDOW_FEATURES.index("left_knee")
# Raptor cache layouts (raptor_data.RAPTOR_VARIANTS); listed here so settings can be
# checked without importing the DICOM preparation code.
RAPTOR_VARIANT_SIZES = {"raptor-widedense-series-safe-v1": 336, "raptor-dense-distinct-v1": 336,
                        "raptor-dense-distinct-448-v1": 448, "raptor-dense-distinct-384-v1": 384,
                        "raptor-dense-distinct-160mm-384-v1": 384, "raptor-dense-distinct-200mm-384-v1": 384,
                        "raptor-dense-distinct-knee-384-v1": 384, "raptor-dense-distinct-knee-closeup-384-v1": 384}
RAPTOR_VARIANT_NAMES = list(RAPTOR_VARIANT_SIZES)
