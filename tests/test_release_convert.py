import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtracer import convert
from xtracer.log import Logger


def close_test_logger():
    for handler in list(Logger.logger.handlers):
        Logger.logger.removeHandler(handler)
        handler.close()


class ConvertReleaseTests(unittest.TestCase):
    def tearDown(self):
        close_test_logger()

    def test_minimal_tdf_schema_is_created_without_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tdf = Path(temp_dir) / 'analysis.tdf'
            db = convert._create_tdf(tdf, 'sample.mbi')
            db.commit()
            db.close()
            check = sqlite3.connect(tdf)
            try:
                tables = {
                    row[0]
                    for row in check.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertTrue({
                    'Frames',
                    'GlobalMetadata',
                    'MzCalibration',
                    'TimsCalibration',
                    'Segments',
                    'DiaFrameMsMsInfo',
                    'DiaFrameMsMsWindowGroups',
                    'DiaFrameMsMsWindows',
                }.issubset(tables))
                metadata = dict(check.execute('SELECT Key, Value FROM GlobalMetadata'))
                self.assertEqual(metadata['AcquisitionSoftwareVendor'], 'xTracer')
                self.assertEqual(metadata['SampleName'], 'sample.mbi')
            finally:
                check.close()

    def test_existing_single_output_is_skipped_without_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'sample.mbi'
            output = root / 'sample.d'
            source.touch()
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            with mock.patch.object(convert, '_convert') as run:
                result = convert.main([str(source), '-o', str(output)])
            self.assertEqual(result, 0)
            run.assert_not_called()
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_force_removes_target_before_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'sample.mbi'
            output = root / 'sample.d'
            source.touch()
            output.mkdir()
            (output / 'old.txt').touch()

            def fake_convert(_source, target, _logger):
                self.assertFalse(target.exists())
                target.mkdir()
                (target / 'new.txt').touch()

            with mock.patch.object(convert, '_convert', side_effect=fake_convert):
                result = convert.main([
                    str(source), '-o', str(output), '--force'
                ])
            self.assertEqual(result, 0)
            self.assertFalse((output / 'old.txt').exists())
            self.assertTrue((output / 'new.txt').exists())

    def test_batch_default_output_folder_is_mbi2d(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'sample.mbi').touch()

            def fake_convert(_source, target, _logger):
                target.mkdir(parents=True)

            with mock.patch.object(convert, '_convert', side_effect=fake_convert):
                result = convert.main(['-ws_in', str(root)])
            self.assertEqual(result, 0)
            self.assertTrue((root / 'mbi2d' / 'sample.d').is_dir())

    def test_batch_uses_one_log_and_one_output_per_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ('b.mbi', 'a.mbi'):
                (root / name).touch()

            def fake_convert(_source, target, _logger):
                target.mkdir(parents=True)

            with mock.patch.object(convert, '_convert', side_effect=fake_convert) as run:
                result = convert.main([
                    '-ws_in', str(root), '-out_name', 'converted'
                ])
            self.assertEqual(result, 0)
            self.assertEqual(run.call_count, 2)
            self.assertTrue((root / 'converted' / 'a.d').is_dir())
            self.assertTrue((root / 'converted' / 'b.d').is_dir())
            logs = list((root / 'converted').glob('xtracer_convert_*.log'))
            self.assertEqual(len(logs), 1)


if __name__ == '__main__':
    unittest.main()
