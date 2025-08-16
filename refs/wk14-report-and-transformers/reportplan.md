plan : un maximum de contributions personnelles
on part du sobolev training, on écrit les théorèmes qui montrent la décomposition du NTK
on décrit le RKHS pour le kernel de sobolev, puis le laplace pour le NTK en 1 theoreme rapide
on décrit la biggest ntk eigenvalue, ensuite on décrit les theoremes de terjek, et le plan de preuve
pour obtenir une vraie borne, on part sur l'agenda complet pour faire de la RMT
le but est d'étudier les statistiques locales de la surface et son scaling sur la width

ensuite on part sur ce que l'on veut, obtenir une finite width corrections, on écrit tous les theoremes (preuves en appendix), tout doti se faire comme un papier
ensuite on décrit les techniques d'approchesde  préconditionnement de NTK, on a le MMNN et transformers ,
décrire ensuite les régimes, feature et kernel
la tactique pour le NTK est de retrouver la gaussianité
puis d'appliquer sur le NNGP la double dérivation comme un price theorem
mettre les gros graphes des expériences numérique 
3 parties : MMNN, FCNN, transformers
on décrit tous les calculs qui mènenet à nos formules, les hypothèses, et ainsi que où nous allons et pourquoi (avec un high level structure en tikz)

la finalité est d'obtenir une borne pour les smallest eigenvalue, donc en obtenir 1, expliquer les 2 approches
avec le khatri rao et retirer les autres termes d'hermite et pourquoi ça ne peut pas fonctionner
la deuxieme approche est l'intuition du 

la preuve a la khatri rao peut etre etendue dans le settings resnet


finaliser sur notre intuition finale qui explique les performances pour les MMNN, transformers et resnets (discussion avec terjek aussi)
ne pas oublier que la suite est longue, en particulier pour le RKHS et 

enfin partir sur l'attaque des FCNN avec la finite width correction, expliciter les théorèmes et formules trouvées ainsi que les experiences sur les scaling
law du reste, et finaliser sur l'intuition finale, les résultats et les discussions sur l'attaque du dernier terme (QUE & GOE)
et pour les MMNN expliquer que l'aspect global de l'optimisation est beaucoup plus dur mais qu'on bosse en ce moment sur ça,
en particulier on conjecture qu'autour d'un minima global, le problème est bien stable est qu'on peut plonger dedans assez bien
(avec des experiences numériques qui le justiifie)

on va aussi rajouter un tracking le long des semaines de tout ce qui a été fait
bien séparer le meanfield regime du kernel regime, et avec le framework TP

faire une revue du framework TP en appendix
opur la délocalisation, on mettra les experiences sur le scaling des vp
faire des dernieres experiences numeriques avec l'approche hessienne du ntk, et des formules et l'approche finale de l'étude du termes à la lyapunov
ensuite expliquer que pour la concentration bounds on utilisera celle de terjek, en particulier avec résseau parabolique, et comme quoi
ce type de réseau permet de se placer dans une feature learning tout en maintenant une ntk-itude par un scaling de la dernière et des derniers papiers
c'est la 2eme approche pour FCNN, donc on en a 2

d'ailleurs expliquer que je bosse en ce moment conjointement avec, en particulier pour la taylor decomposition
et expliquer que pour les mmnns on récupère 

parler de l'application en sciml surtout et de ce que l'on veut observe pour les grands modèles (ou de fondations ) en sci ml ou non
et le scaling des parameters, est ce que les hypotheses theoriqeus sont verifiées etc .. 
parler de l'aspect phystat avec boltzmann machine, et convergence à l'equilibre thermique

(voir soutenance de these typique comment ça se passe)
surtout parler de l'angle phystat pour la soutenance parce que le jury est calé en ça 
soutenance : on parlera surtout des aspects pratiques et de comment voir les théorèmes sous la lumière (et interpréter les expériences numériques)
choisir les bons graphes à mettre, rassifier le github, pour avoir des runs et des notebooks rapides et parfaits, et pour run sur cpu aussi peut etre, avec un bon readme
laisser mes refs seulement dans une branche
insister sur la reproducibilité
expliquer aussi que j'ai tout fait en live pendant beaucoup de temps, (meme l'ecriture de ceci)

rassifier youtube creator studio