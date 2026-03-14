import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from batterymat.screening_cathode.dft_prep import (
    compute_dft_voltage,
    write_pbe_relax_incar,
    write_tmbj_static_incar,
    write_kpoints,
    write_potcar_spec,
    write_tmbj_readme,
    get_delithiated_structure,
    generate_inputs,
)


def test_compute_dft_voltage_positive_for_cathode():
    # Cathode: delithiated structure has higher energy than lithiated
    # V = (E_delith - E_lith - n_li * E_li_metal) / n_li
    # e_delithiated = -8.0 + 2.0 = -6.0
    # V = (-6.0 - (-8.0) - 1 * (-1.90)) / 1 = (2.0 + 1.90) = 3.90
    v = compute_dft_voltage(
        e_lithiated=-8.0,
        e_delithiated=-6.0,
        n_li=1,
        e_li_metal=-1.90,
    )
    assert abs(v - 3.90) < 1e-6


def test_compute_dft_voltage_formula():
    v = compute_dft_voltage(
        e_lithiated=-100.0,
        e_delithiated=-95.0,
        n_li=2,
        e_li_metal=-2.0,
    )
    # V = ((-95) - (-100) - 2*(-2)) / 2 = (5 + 4) / 2 = 4.5
    assert abs(v - 4.5) < 1e-6


def test_compute_dft_voltage_raises_on_zero_n_li():
    with pytest.raises(ValueError, match="n_li"):
        compute_dft_voltage(-100.0, -95.0, 0, -1.90)


def test_write_pbe_incar_required_tags(tmp_path):
    write_pbe_relax_incar(tmp_path)
    incar = (tmp_path / "INCAR").read_text()
    for tag in ["IBRION = 2", "NSW = 100", "ISIF = 3", "ENCUT = 520",
                "EDIFF = 1E-6", "EDIFFG = -0.01", "LCHARG = .TRUE."]:
        assert tag in incar, f"Missing tag: {tag}"


def test_write_tmbj_incar_required_tags(tmp_path):
    write_tmbj_static_incar(tmp_path)
    incar = (tmp_path / "INCAR").read_text()
    for tag in ["IBRION = -1", "NSW = 0", "METAGGA = MBJ",
                "LASPH = .TRUE.", "ICHARG = 11", "LORBIT = 11",
                "NEDOS = 2000"]:
        assert tag in incar, f"Missing tag: {tag}"


def test_write_kpoints_creates_file(tmp_path):
    write_kpoints(tmp_path)
    kpoints = (tmp_path / "KPOINTS").read_text()
    assert "Gamma" in kpoints or "gamma" in kpoints.lower()


def test_write_potcar_spec_known_element(tmp_path):
    write_potcar_spec(tmp_path, ["Li", "Mn", "O"])
    spec = (tmp_path / "POTCAR_spec").read_text()
    assert "Li_sv" in spec
    assert "Mn_pv" in spec
    assert "O" in spec


def test_write_potcar_spec_unknown_element_fallback(tmp_path):
    write_potcar_spec(tmp_path, ["Li", "Xx"])  # Xx is not in the table
    spec = (tmp_path / "POTCAR_spec").read_text()
    assert "Xx" in spec  # bare symbol used as fallback


def test_write_tmbj_readme_mentions_chgcar(tmp_path):
    pbe_dir = tmp_path / "pbe_run"
    pbe_dir.mkdir()
    write_tmbj_readme(tmp_path, pbe_dir)
    readme = (tmp_path / "README").read_text()
    assert "CHGCAR" in readme
    assert str(pbe_dir) in readme


def _make_mock_atoms(elements):
    """Create a minimal mock Atoms-like object."""
    mock = MagicMock()
    mock.elements = elements
    mock.num_atoms = len(elements)
    return mock


def test_get_delithiated_raises_on_no_li():
    atoms = _make_mock_atoms(["Mn", "O", "O"])
    with pytest.raises(ValueError, match="no Li atoms"):
        get_delithiated_structure(atoms)


def test_get_delithiated_single_li_skips_vacancy_enumeration():
    """When n_Li == 1, return the host structure directly (no Vacancy call)."""
    # Build a real minimal Atoms via jarvis if available, else skip
    pytest.importorskip("jarvis")
    from jarvis.core.atoms import Atoms

    # Simple LiMnO2-like structure: 1 Li + host
    atoms = Atoms(
        lattice_mat=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        elements=["Li", "Mn", "O", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        cartesian=False,
    )
    with patch("batterymat.screening_cathode.dft_prep.Vacancy") as mock_vac:
        result = get_delithiated_structure(atoms)
        mock_vac.assert_not_called()
    assert "Li" not in result.elements


def test_get_delithiated_multi_li_uses_alignn_ranking():
    """When n_Li > 1, calls Vacancy.generate_defects and ranks by ALIGNN energy."""
    pytest.importorskip("jarvis")
    from jarvis.core.atoms import Atoms

    atoms = Atoms(
        lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        elements=["Li", "Li", "Mn", "O", "O", "O", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
        ],
        cartesian=False,
    )

    # Mock Vacancy to return two fake defect structures
    mock_defect_1 = MagicMock()
    mock_defect_1._defect_structure = MagicMock()
    mock_defect_1._symbol = "Li"
    mock_defect_2 = MagicMock()
    mock_defect_2._defect_structure = MagicMock()
    mock_defect_2._symbol = "Li"

    # Mock ALIGNN to return energies: defect_1 = -10.0, defect_2 = -12.0
    # Lower energy -> defect_2 should be selected
    def mock_energy(defect_atoms):
        if defect_atoms is mock_defect_1._defect_structure:
            return -10.0
        return -12.0

    with patch("batterymat.screening_cathode.dft_prep.Vacancy") as MockVac, \
         patch("batterymat.screening_cathode.dft_prep._alignn_energy", side_effect=mock_energy):
        instance = MockVac.return_value
        instance.generate_defects.return_value = [mock_defect_1, mock_defect_2]
        result = get_delithiated_structure(atoms)

    assert result is mock_defect_2._defect_structure


def _make_mock_dft3d(jid, elements):
    """Return a minimal fake dft3d DataFrame row."""
    import pandas as pd
    from jarvis.core.atoms import Atoms
    atoms = Atoms(
        lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        elements=elements,
        coords=[[float(i) / len(elements)] * 3 for i in range(len(elements))],
        cartesian=False,
    )
    return pd.DataFrame([{"jid": jid, "atoms": atoms.to_dict()}])


def test_generate_inputs_creates_four_dirs(tmp_path):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-999", ["Li", "Mn", "O", "O"])
    with patch("batterymat.screening_cathode.dft_prep.get_delithiated_structure") as mock_delith:
        from jarvis.core.atoms import Atoms
        mock_delith.return_value = Atoms(
            lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            elements=["Mn", "O", "O"],
            coords=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cartesian=False,
        )
        created = generate_inputs(["JVASP-999"], output_dir=str(tmp_path), dft3d_df=dft3d)

    jid_dir = tmp_path / "JVASP-999"
    assert jid_dir.exists()
    for sub in ["1_pbe_relax_lithiated", "2_pbe_relax_delithiated",
                "3_tmbj_static_lithiated", "4_tmbj_static_delithiated"]:
        assert (jid_dir / sub).exists(), f"Missing: {sub}"

    assert len(created) == 1
    assert created[0] == str(jid_dir)


def test_generate_inputs_each_dir_has_required_files(tmp_path):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-888", ["Li", "Co", "O", "O"])
    with patch("batterymat.screening_cathode.dft_prep.get_delithiated_structure") as mock_delith:
        from jarvis.core.atoms import Atoms
        mock_delith.return_value = Atoms(
            lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            elements=["Co", "O", "O"],
            coords=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cartesian=False,
        )
        generate_inputs(["JVASP-888"], output_dir=str(tmp_path), dft3d_df=dft3d)

    jid_dir = tmp_path / "JVASP-888"
    for sub in ["1_pbe_relax_lithiated", "2_pbe_relax_delithiated"]:
        for f in ["POSCAR", "INCAR", "KPOINTS", "POTCAR_spec"]:
            assert (jid_dir / sub / f).exists(), f"Missing {sub}/{f}"
    for sub in ["3_tmbj_static_lithiated", "4_tmbj_static_delithiated"]:
        for f in ["POSCAR", "INCAR", "KPOINTS", "POTCAR_spec", "README"]:
            assert (jid_dir / sub / f).exists(), f"Missing {sub}/{f}"


def test_generate_inputs_skips_zero_li(tmp_path, capsys):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-777", ["Mn", "O", "O"])  # No Li
    created = generate_inputs(["JVASP-777"], output_dir=str(tmp_path), dft3d_df=dft3d)
    captured = capsys.readouterr()
    assert len(created) == 0
    assert "WARNING" in captured.out
    assert "JVASP-777" in captured.out


def test_generate_inputs_returns_list_of_paths(tmp_path):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-555", ["Li", "V", "O", "O", "O", "O", "O"])
    with patch("batterymat.screening_cathode.dft_prep.get_delithiated_structure") as mock_delith:
        from jarvis.core.atoms import Atoms
        mock_delith.return_value = Atoms(
            lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            elements=["V", "O", "O"],
            coords=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cartesian=False,
        )
        result = generate_inputs(["JVASP-555"], output_dir=str(tmp_path), dft3d_df=dft3d)
    assert isinstance(result, list)
    assert len(result) == 1
