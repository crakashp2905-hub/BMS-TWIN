import unittest
from btms_core.contracts.types import *
from btms_core.validation.validator import validate_packet
from btms_core.parameters.registry import *
from btms_core.heat.ledger import *

class CoreTests(unittest.TestCase):
    def packet(self, unit="V", timestamp=10.0):
        s=Signal("cell_voltage",3.2,unit,timestamp,"sensor",SourceType.MEASURED)
        return BatteryStatePacket("cell","LFP_GRAPHITE",10.0,{"cell_voltage":s})
    def test_valid_packet(self):
        r=validate_packet(self.packet(),[RequiredInput("cell_voltage","V")],1.0)
        self.assertEqual(r.sufficiency,DataSufficiency.SUFFICIENT)
    def test_unit_mismatch(self):
        r=validate_packet(self.packet("mV"),[RequiredInput("cell_voltage","V")])
        self.assertEqual(r.sufficiency,DataSufficiency.INVALID)
    def test_stale_packet(self):
        r=validate_packet(self.packet(timestamp=1.0),[RequiredInput("cell_voltage","V")],2.0)
        self.assertEqual(r.sufficiency,DataSufficiency.STALE)
    def test_missing_input(self):
        p=BatteryStatePacket("cell","LFP_GRAPHITE",1.0,{})
        r=validate_packet(p,[RequiredInput("cell_voltage","V")])
        self.assertEqual(r.sufficiency,DataSufficiency.INSUFFICIENT)
    def test_parameter_unit_guard(self):
        reg=ParameterRegistry()
        reg.add(ParameterRecord("capacity",2.3,"Ah","LFP","provisional",ParameterStatus.PROVISIONAL))
        with self.assertRaises(ValueError): reg.get_value("LFP","capacity","C")
    def test_heat_owner_duplicate_guard(self):
        ledger=HeatLedger()
        ledger.add(HeatContribution("a","short_path","MICRO_SHORT",0.1,HeatBasis.LOCAL_DEFECT,incremental_abnormal=True))
        with self.assertRaises(ValueError):
            ledger.add(HeatContribution("b","short_path","OTHER",0.2,HeatBasis.LOCAL_DEFECT,incremental_abnormal=True))
        self.assertAlmostEqual(ledger.total_incremental_abnormal_power_w(),0.1)

if __name__=="__main__": unittest.main()
