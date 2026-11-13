"""Canonical input contract for direct internal-short-path measurements."""
from btms_core.contracts.types import RequiredInput, SourceType

# These are deliberately NOT ordinary cell-terminal measurements. They must
# represent voltage across, and current through, the suspected conductive path.
REQUIRED_INPUTS = (
    RequiredInput(
        name="isc_path_voltage_v", unit="V",
        description="Voltage directly across the identified internal conductive path.",
        source_types_allowed=(SourceType.MEASURED, SourceType.ESTIMATED),
    ),
    RequiredInput(
        name="isc_path_current_a", unit="A",
        description="Current directly through the same identified conductive path; positive means dissipative path current.",
        source_types_allowed=(SourceType.MEASURED, SourceType.ESTIMATED),
    ),
)

OPTIONAL_INPUTS = (
    RequiredInput(
        name="isc_effective_volume_m3", unit="m^3", required=False,
        description="Validated effective volume over which path heat is distributed; not the whole cell volume by default.",
        source_types_allowed=(SourceType.MEASURED, SourceType.CONFIGURED, SourceType.ESTIMATED),
    ),
    RequiredInput(
        name="isc_observation_duration_s", unit="s", required=False,
        description="Duration for which the measured path power can be assumed representative.",
        source_types_allowed=(SourceType.MEASURED, SourceType.CONFIGURED, SourceType.ESTIMATED),
    ),
)
