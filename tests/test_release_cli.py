import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from xtracer import gui
from xtracer import main as cli
from xtracer.log import Logger
from xtracer.search import format_mgf_peaks


def close_test_logger():
    for handler in list(Logger.logger.handlers):
        Logger.logger.removeHandler(handler)
        handler.close()


class CommandLineReleaseTests(unittest.TestCase):
    def tearDown(self):
        close_test_logger()

    def test_root_help_lists_only_command_family(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(['--help']), 0)
        help_text = stdout.getvalue()
        self.assertIn('{search,convert,gui}', help_text)
        self.assertIn('search', help_text)
        self.assertIn('convert', help_text)
        self.assertIn('gui', help_text)

    def test_legacy_direct_search_arguments_are_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli.main(['-ws_in', '.', '-xix'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('unknown command', stderr.getvalue())

    def test_each_command_has_its_own_help(self):
        for command, marker in (
            ('search', 'Generate PAMAF pseudo-spectra'),
            ('convert', 'Convert PAMAF .mbi files'),
            ('gui', 'Launch the xTracer viewer'),
        ):
            stdout = io.StringIO()
            with self.subTest(command=command):
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    cli.main([command, '--help'])
                self.assertEqual(raised.exception.code, 0)
                self.assertIn(marker, stdout.getvalue())

    def test_search_defaults_and_empty_input_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = cli.parse_search_args(['-ws_in', temp_dir, '-xix'])
            self.assertEqual(args.tol_iso_num, 2)
            self.assertEqual(args.tol_pcc, 0.3)
            self.assertEqual(args.xic_across_cycle_num, 7)
            self.assertEqual(cli.run_search(['-ws_in', temp_dir, '-xix']), 2)
            logs = list((Path(temp_dir) / 'mgf_xtracer').glob('xtracer_search_*.log'))
            self.assertEqual(len(logs), 1)
            self.assertIn('no .mbi files found', logs[0].read_text(encoding='utf-8'))

    def test_write_pcc_peak_block_is_bytes(self):
        block = format_mgf_peaks(
            np.array([100.1234567]),
            np.array([42.125]),
            np.array([0.876]),
        )
        self.assertIsInstance(block, bytes)
        self.assertEqual(block, b'100.123457 42.12 0.88\n')
        self.assertEqual(block + b'END IONS\n\n', b'100.123457 42.12 0.88\nEND IONS\n\n')

    def test_gui_opens_browser_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mbi = root / 'sample.mbi'
            mgf = root / 'sample.mgf'
            sage = root / 'results.sage.tsv'
            matched = root / 'matched_fragments.sage.tsv'
            for path in (mbi, mgf, sage, matched):
                path.touch()
            with mock.patch.object(
                subprocess,
                'run',
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                result = gui.main([
                    '--mbi', str(mbi),
                    '--mgf', str(mgf),
                    '--sage-results', str(sage),
                    '--out-dir', str(root / 'logs'),
                ])
            self.assertEqual(result, 0)
            command = run.call_args.args[0]
            headless_index = command.index('--server.headless')
            self.assertEqual(command[headless_index + 1], 'false')
            self.assertEqual(Path(command[-1]), matched.resolve())

    def test_gui_ctrl_c_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / name for name in (
                'sample.mbi', 'sample.mgf', 'results.sage.tsv', 'matched_fragments.sage.tsv'
            )]
            for path in paths:
                path.touch()
            with mock.patch.object(subprocess, 'run', side_effect=KeyboardInterrupt):
                result = gui.main([
                    '--mbi', str(paths[0]),
                    '--mgf', str(paths[1]),
                    '--sage-results', str(paths[2]),
                    '--out-dir', str(root / 'logs'),
                ])
            self.assertEqual(result, 0)

    def test_gui_normalizes_url_encoded_sage_filename(self):
        self.assertEqual(
            gui.normalize_sage_filename('sample%20name.mgf'),
            'sample name.mgf',
        )


if __name__ == '__main__':
    unittest.main()
