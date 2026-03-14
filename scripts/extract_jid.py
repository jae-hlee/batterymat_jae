input_text = """jid  formula Spg SpgNum  crys    func    E_form  OPT_gap MBJ_gap hse_gap Kv  Gv  poisson spillage    slme    mag type
JVASP-18433 LiNiO2  R-3m    166 trigonal    OptB88vdW   -1.44752    0.000   0.516   -   135.52  54.73   0.33    -   16.07   1.0 3D
JVASP-52789 Li2NiO3 C2/c    15  monoclinic  OptB88vdW   -1.65825    1.064   -   -   -   -   -   -   -   0.0 3D
JVASP-42629 Li3Ni4O8    C2/m    12  monoclinic  OptB88vdW   -1.34463    0.000   -   -   -   -   -   -   -   2.971   3D
JVASP-85546 LiNiO2  P6/mmm  191 hexagonal   OptB88vdW   -0.67338    0.000   -   -   -   -   -   -   -   6.512   3D
JVASP-108189    Li2NiO2 R-3m    166 trigonal    OptB88vdW   -1.45273    0.975   -   -   -   -   -   -   -   2.0 3D
JVASP-100060    LiNiO2  Imm2    44  orthorhombic    OptB88vdW   -1.39395    0.000   -   -   98.99   41.15   0.36    -   -   0.914   3D
JVASP-94274 LiNiO2  I4_1/amd    141 tetragonal  OptB88vdW   -1.47080    0.000   -   -   -   -   -   -   -   0.0 3D
JVASP-103750    LiNi4O5 I4/m    87  tetragonal  OptB88vdW   -1.02859    0.000   -   -   -   -   -   -   -   0.0 3D
JVASP-8375  LiNi2O4 P2/m    10  monoclinic  OptB88vdW   -1.21053    0.000   0.522   -   109.92  47.71   0.31    -   16.32   0.982   3D
JVASP-8644  LiNiO2  Cm  8   monoclinic  OptB88vdW   -1.40939    0.000   0.381   -   126.51  32.39   0.42    -   9.08    0.999   3D
JVASP-7881  Li2NiO2 Immm    71  orthorhombic    OptB88vdW   -1.67456    0.580   -   -   110.27  61.68   0.28    -   -   0.0 3D
JVASP-8646  LiNiO2  Imm2    44  orthorhombic    OptB88vdW   -1.39380    0.000   0.0 -   99.22   40.95   0.35    -   -   0.914   3D
JVASP-51355 LiNi2O4 Fd-3m   227 cubic   OptB88vdW   -1.26678    0.000   -   -   -   -   -   -   -   1.808   3D
JVASP-79795 LiNiO2  R-3m    166 trigonal    OptB88vdW   -1.44738    0.000   -   -   -   -   -   -   -   1.0 3D
JVASP-79748 LiNiO2  Cm  8   monoclinic  OptB88vdW   -1.40942    0.000   -   -   -   -   -   -   -   0.999   3D
JVASP-30393 Li2NiO2 R-3m    166 trigonal    OptB88vdW   -1.28228    0.000   0.0 -   -   -   -   -   -   -   3D
JVASP-45183 LiNi2O4 P2/m    10  monoclinic  OptB88vdW   -1.21045    0.000   -   -   111.04  46.55   0.31    -   -   0.98    3D
JVASP-44761 Li3NiO3 P4_2/mnm    136 tetragonal  OptB88vdW   -1.72376    0.000   -   -   -   -   -   -   -   3.976   3D
JVASP-44653 Li7Ni5O12   C2/m    12  monoclinic  OptB88vdW   -1.55358    0.000   -   -   -   -   -   -   -   2.926   3D
JVASP-12216 Li2NiO2 P-3m1   164 trigonal    OptB88vdW   -1.51564    1.094   -   -   -   -   -   -   -   4.0 3D
JVASP-57390 Li2NiO3 C2/m    12  monoclinic  OptB88vdW   -1.65832    1.023   -   -   125.28  83.09   0.23    -   -   0.0 3D
JVASP-110313    LiNiO2  R-3m    166 trigonal    OptB88vdW   -1.47202    0.000   -   -   -   -   -   -   -   1.0 3D
JVASP-116288    LiNiO2  R-3m    166 trigonal    OptB88vdW   -1.46188    0.000   -   -   -   -   -   -   -   1.0 3D
JVASP-112758    Li3Ni4O8    C2/m    12  monoclinic  OptB88vdW   -1.36896    0.000   -   -   -   -   -   -   -   2.74    3D
JVASP-112795    Li2NiO3 C2/c    15  monoclinic  OptB88vdW   -1.68484    1.007   -   -   -   -   -   -   -   0.0 3D
JVASP-111149    LiNiO2  C2/m    12  monoclinic  OptB88vdW   -1.47116    0.000   -   -   -   -   -   -   -   1.999   3D
JVASP-117271    LiNi2O4 Imma    74  orthorhombic    OptB88vdW   -1.22626    0.000   -   -   -   -   -   -   -   1.941   3D
JVASP-119319    LiNiO2  C2/m    12  monoclinic  OptB88vdW   -1.47451    0.000   -   -   -   -   -   -   -   3.997   3D
JVASP-116855    Li7Ni5O12   P-1 2   triclinic   OptB88vdW   -1.57219    0.000   -   -   -   -   -   -   -   2.907   3D
JVASP-118606    Li2NiO2 Immm    71  orthorhombic    OptB88vdW   -1.69478    0.524   -   -   -   -   -   -   -   0.0 3D
JVASP-122334    Li7Ni5O12   C2/m    12  monoclinic  OptB88vdW   -1.57820    0.000   -   -   -   -   -   -   -   2.921   3D
JVASP-140606    Li2Ni5O7    C2/m    12  monoclinic  OptB88vdW   -1.18050    0.000   -   -   -   -   -   -   -   0.819   3D
JVASP-140204    LiNi2O4 Fd-3m   227 cubic   OptB88vdW   -1.28460    0.000   -   -   -   -   -   -   -   1.749   3D
JVASP-141853    Li2Ni3O6    P-1 2   triclinic   OptB88vdW   -1.32198    0.000   -   -   -   -   -   -   -   3.78    3D
JVASP-132936    LiNi5O10    P-1 2   triclinic   OptB88vdW   -0.99448    0.000   -   -   -   -   -   -   -   0.0 3D
JVASP-144724    Li3Ni4O8    C2/m    12  monoclinic  OptB88vdW   -1.36924    0.000   -   -   -   -   -   -   -   2.738   3D
JVASP-144040    Li5Ni7O12   C2  5   monoclinic  OptB88vdW   -1.37783    0.000   -   -   -   -   -   -   -   2.417   3D
JVASP-141005    Li3Ni4O8    R-3 148 trigonal    OptB88vdW   -1.37005    0.000   -   -   -   -   -   -   -   2.979   3D
JVASP-133664    Li7Ni5O12   C2  5   monoclinic  OptB88vdW   -1.52766    0.000   -   -   -   -   -   -   -   0.0 3D
JVASP-142650    LiNi2O3 C2/c    15  monoclinic  OptB88vdW   -1.26150    0.000   -   -   -   -   -   -   -   5.243   3D
JVASP-146919    Li3Ni4O8    R-3 148 trigonal    OptB88vdW   -1.37014    0.000   -   -   -   -   -   -   -   2.98    3D
JVASP-145597    LiNi2O4 P6_3mc  186 hexagonal   OptB88vdW   -1.25375    0.000   -   -   -   -   -   -   -   3.013   3D
JVASP-77996 Li2NiO2 P-3m1   164 trigonal    OptB88vdW   -1.39879    0.961   -   -   -   -   -   -   -   2.001   2D"""

# Split into lines, skip the header
lines = input_text.strip().split("\n")[1:]

# Extract only the jid (first word in each line)
jids = [line.split()[0] for line in lines if line.strip()]

# Save to a file
with open("jid_list.txt", "w") as f:
        for jid in jids:
                    f.write(jid + "\n")
