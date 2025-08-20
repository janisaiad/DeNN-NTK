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


bien garder à l'esprit que l'on garde en vue le fait que dans les applications indusrie, ce n'est pas forcément la meilleure chose
que de trouver la meilleure approximation (les théoremes d'approximations on s'en fout un peu)
c'est surtout la généralisation qui intéresse et l'optimization en 1er lieu

pour la wk 3 kernel integral operator à clarifier
et bien expliciter que l'on travaille sur 2 domaines et qu'en sciml/optimization les 2 sont importants, que ce soit 
pour le rkhs ou pour l'optimization bounds
un aspect super important est qu'il faut guider vers la comprehension de deeper or wider pour nn regression
et qu'on repond d'abord pour nn regression, ensuite pour sobolev avec la decomposition
le fait est qu'en parallele on apprend aussi à fitter nos données et leurs normes, ce n'est pas la 1ere chose que l'on fait
(car on a une matrice diagonale), c'est pas on apprend 2 noyaux orthogonaux, on apprend en fait 1 noyaux et son scaling


mettre aussi l'aspect ingéniérie et le fait que l'asd et le ntk est dual, et que ça explique en partie la double descente
il existe d'autres explications rigoureuses pour une regression L2, l'asd a des eigenvalues nulles et donc
une partie de l'espace des parametres ne sert à rien
et d'ailleurs le scaling de ces eigenvalue est exponentiel avec un gros bulk, ce qui odnne
et d'ailleurs on peut calculer ces eigenvalues (en L2 par exemple pour moindres carrés)
et ça dépend des données, selon un noyau, qui est celui de laplace ! donc 
c'est du point de vue du RKHS qu'il faut se baser pour expliquer la double descente (comme en moindre carrés) !! donc c'est cohérent

les poids apprennent une representation sacling independantes en meme temps que le scaling (juste en divisant)
les eigenvectors de AU sont, la bihomogeneite ne change pas les directoins
mais relativement aux autres, un point vraiment loin va etre mal appris, car la matrice diagonale va avoir N-1 eigenvalue positives
(la derniere tend vers 0) et donc on l'apprend, c'est vraiment la norme relative qui importe
et ça ça explique ce qu'on voit quand on train une fonction sinus, c'est un biais inductif qui fait apprendre d'abord 0, puis les entrées
proches de 0 etc .. 


aussi on ne se focus pas sur l'ERM, on se focus pas sur identifier qu'il est mais comment le trouver avec des méthodes d'apprentissage

expliciter l'eatpe douteuse de retirer tous les termes de legendre ( sinon on suppose la polynoamialité ce qui n'est vraiment pas bon quand on stack les layers)
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