# BatteryMat: Computational Design Pipeline for Battery Materials


Start with Google colab

https://colab.research.google.com/gist/knc6/e7e3aecf3b748c0df1d58dc7911da950/umlff_comparison_batterymat.ipynb


https://docs.google.com/spreadsheets/d/19CBtfdQxwpG8cIty31UYuL5xp8HAd2Ik/edit?gid=1360471168#gid=1360471168



GDrive link: https://drive.google.com/drive/u/2/folders/1vf42QPRKc_kFXo5kmnWvDKzRFErUDdU3

https://github.com/ndrewwang/BotB/blob/main/1.%20BotB%20Theoretical%20Capacities.ipynb

## Stage-1: Alternative anode materials (voltage calc)

 - Literature search (Michael)
 - Volume relaxation, large volume change not recommended (threshold: below 20 %??), also updated voltage profile after relaxation
 - 1) number of elements in structure, 2) density, 3) space group or crystal symmetry, 4) capacity, 5) average voltage

## Stage-2: NEB barrier calculation

  - Given ion/wyckoff position and crystal, generate all the images (Kamal) [2x2x2 in general]
  - Henkelmann script (Michael)

## Stage-3: Benchmarking

   - Currently voltage profile for NMC, need 10+ more
   - NEB barriers for V2O5, TiSe2, need a few more
   - Chevrel phase (Mo6S8)
   - MnO2
   - Review papers with actual data (? Michael+Smeu)
   - Battery explorer MP (Michael)

## Stage-4: Experimental synthesis

## References:

1. [An Overview and Future Perspectives of Aluminum Batteries](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.201601357)
2. [First principles investigation into the interwoven nature of voltage and mechanical properties of the Li NMC-811 cathode](https://www.sciencedirect.com/science/article/pii/S0378775321011162)
3. [TiSe2 cathode for beyond Li-ion batteries](https://www.sciencedirect.com/science/article/pii/S0378775319307980?via%3Dihub)
4. [Effects of Impurities on the Electrochemical Properties of LiCoO2](https://iopscience.iop.org/article/10.1149/1.2220905)
5. [Understanding the Irreversible Reaction Pathway of the Sacrificial Cathode Additive Li6CoO4](https://onlinelibrary.wiley.com/doi/full/10.1002/aenm.202301132)
6. [Modeling the Voltage Profile for LiFePO4](https://iopscience.iop.org/article/10.1149/1.2006607)
7. [Studies of Spinel-to-Layered Structural Transformations in LiMn2O4 Electrodes Charged to High Voltages](https://pubs.acs.org/doi/10.1021/acs.jpcc.7b00929)
8. [Ab initio investigation of α- and ζ-V2O5 for beyond lithium ion battery cathodes](https://doi.org/10.1016/j.jpowsour.2020.228096)
9. [Hybrid density functional theory modeling of Ca,Zn, and Al ion batteries using the Chevrel phase Mo6S8 cathode](https://doi.org/10.1039/c7cp03378h)
10. [Density Functional Theory Modeling of MnO2Polymorphs asCathodes for Multivalent Ion Batteries](https://doi.org/10.1021/acs.jpcc.8b00918)
11. [Tailoring the Morphology of LiCoO 2: A First Principles Study](https://pubs.acs.org/doi/epdf/10.1021/cm9008943)
12. [Ab initio calculation of the intercalation voltage of lithium-transition-metal oxide electrodes for rechargeable batteries](https://ceder.berkeley.edu/publications/jps-68-664-1997.pdf)

    (old) https://colab.research.google.com/gist/knc6/dfe66eb5033b006ca12a8ade2478bf7f/umlff_comparison_batterymat.ipynb
