# Projet_Image

1) Expliquer l'OT entre des GMM
2) Cas n°1: transport vers une GMM en dimension 2
3) Cas n°2: transport vers une distribution qui n'est pas une GMM en dimension 2
4) Cas n°3: transport vers une distribution qui n'est pas une GMM en plus grande dimension (MNIST par ex)

Méthode de résolution d'un cas:
 1) Fitter une GMM aux données <br>
 2) Résoudre l'équation (4.4) du papier pour trouver $w^{*}$
 3) En déduire $T_{\text{mean}}$ en utilisant l'équation de la section 6.3 du papier
 4) Sample des points selon $\mu_{0}$ et utiliser $T_{\text{mean}}$
