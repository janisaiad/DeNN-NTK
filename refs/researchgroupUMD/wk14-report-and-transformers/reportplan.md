revoir la base de données de rapport de l'x pour d'inspirer

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


IMPORTANT : un aspect super important est qu'il faut guider vers la comprehension de deeper or wider pour nn regression
et qu'on repond d'abord pour nn regression, ensuite pour sobolev avec la decomposition

IMPORTANT : le fait est qu'en parallele on apprend aussi à fitter nos données et leurs normes, ce n'est pas la 1ere chose que l'on fait
(car on a une matrice diagonale), c'est pas on apprend 2 noyaux orthogonaux, on apprend en fait 1 noyaux et son scaling

expliciter aussi que l'usage de l'IA s'est restreint le plus possible lors de l'écriture de tex, et quand je me suis retrouvé bloqué et que
mon prof n'était pas dispo pour discuter
compiler toutes les reserch directions au fil du temps, et faireun graphe de tout ce que j'ai travaillé, une sorte de graphe du début avec tous les chemins, et une chronologie
avec meme un historique de toutes mes requetes IA mis dans un ficheirs special
détailler aussi tout le setup experimental et mis en oeuvre avec toutes les stats de commit et de fichiers
décrire aussi que j'ai fait des lives, contacts etc ..
mettre aussi l'aspect ingéniérie et le fait que l'asd et le ntk est dual, et que ça explique en partie la double descente
il existe d'autres explications rigoureuses pour une regression L2, l'asd a des eigenvalues nulles et donc
une partie de l'espace des parametres ne sert à rien
et d'ailleurs le scaling de ces eigenvalue est exponentiel avec un gros bulk, ce qui odnne
et d'ailleurs on peut calculer ces eigenvalues (en L2 par exemple pour moindres carrés)
et ça dépend des données, selon un noyau, qui est celui de laplace ! donc 
c'est du point de vue du RKHS qu'il faut se baser pour expliquer la double descente (comme en moindre carrés) !! donc c'est cohérent


il y a beaucoup de choses à récupérer dans les beamers j
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

rassifier youtube creator studio, sur chaque video mettre ce que je fais avec timecode

be careful, the orthogonality under sampling measure is orthogonal in mean ! that is the ntk matrix is random, and that's when we take the 
remarque sur le fait qu'avoir des senseurs sur chebyshev est quelque chose de très très courant, on maitrise cette partie là

ce qui reste )à faire : rederive NTK for tranformers ultra rapidos, et theorem pour mmnn
rajouter une partie explicite sur le fait que noter graphe (en tikz) peut avoir plusieurs minima globaux, et que le ntk garantit que l'on verge bien vers lui

faut pas oublier la condition de dérivée seconde nulle qui est très douteuse et en fait on s'appuie sur un théorèmedeep pour le trianing, et à l'initialisation de mesure nulle
pour le scaling faudra bien rapport les preuves aussi car ce n'est pas forcément giga clair cette linéarité de K3 ou quadratique de K4
pour la soutenance on parlera surtout de physique https://arxiv.org/pdf/2008.08601


evaluation et declaration de plagiat

be careful with d/2
bien détailler la def des chebyshev points, aussi pour c_i=1/n
provide proof in appendices
à la fin faire une revue de tous les théorèmes et preuves

faire bien attention à l'initialisation de A avec ou pas sigma et l'influence sur le NTK
faire une revue de littérature en appendix et sur internet
analyse du x12(2015) : acknowledgements, resume & abstract
intro avec images très générales, et declaration
option, champ, directeurs, d'option et de stage, dates, nom de l'organisme


concise, explain also that this exlpain that we can't achieve to train very deep networks, because under an optimization perspective the problem is ill conditionned with huge alpha


put that : 
\begin{figure}[h!]
    \centering
    % Placeholder for image
    \caption{Characterization of the largest eigenvalue, showing its isolation from the rest of the spectrum.}
    \label{fig:wk7_eig1}
\end{figure}

\begin{figure}[h!]
    \centering
    % Placeholder for image
    \caption{Uniformity test for the bulk eigenvectors, confirming their quasi-random nature.}
    \label{fig:wk7_uniformity}
\end{figure}

\begin{figure}[h!]
    \centering
    % Placeholder for image
    \caption{Empirical spectral distribution of the NTK, illustrating the clear separation between the dominant eigenvalue and the bulk.}
    \label{fig:wk7_spectrum_dist}
\end{figure}

.

























soutenance : 
expliquer ce qui fait changer alpha dan sla soutenance
commencer par parler des 3 refs initiales
mettre la remarque sur les chemins de local optimization du prof sur les slides en tant que conjecture final et chemin à donner
les correlations de misof sont les memes pour O3 et O4
weight decay is also a preconditionning
ntk beta rope aussi
expliquer gaussian corrections for feynman 
faire chronologie du travail
ajoute rune partie asd et fisher
disentangle le papier sur relu non finite width corrections
bien parler du papier fisher etc .. dire que ce travail étant très recent, on l'a rajouté en biblio
mais je n'en parle qu'à l'oral
l'importance de la biblio
toutes les notes de stage que j'ai prise
expliquer à l'oral la technique du 12.1
pg 70  attention nombre de points 
montrer une gradient descent de mmnn
non uniform distribution you should cholesky the convolution operator

faire et montrer un agenda complet de lecture

ce que je retiens aussi pour les deadlines et le formattage, latex, j'ai 4 papiers en cours, donc maintenant j'ai grandi et pris en maturité sur latex,
overleaf et toute la collaboration scientifique
lister tous les gens avec qui j'ai collaboré

recontacter terjek
pris en maturité sur comment gérer de la biblio

presenter le papier feynamal dire c'est quoi ces conneries de tenseurs

la grande lecon c'est qu'il faut tout lire, vraiment tout, la revue la plus grande possible
faire le listing de totutes les personnes et contact, et les contacter
le probleme majeur c'est le temps, et savoir ou trouver le probleme le plus triaivial à resoudr een  1er
choisir à chaque fois le projet pour les resultats
faire

maybe weyl is not tigh 
linear width in transformers
relation de recurrence dans misof et etre sur de son exposant

omg le nouvel article
à la fin analyse directe de muon et adam


















qu'est ce que l'on retient pour l'oral : 20 min 15 slides

- le début, 3 papier, méthodologies, détails de comment s'est déroulé le stage en pratique



Le point de départ c'est 2 choses, 1 papier d'openAI qui a introduit les scaling law

Au delà d'avoir 100k gpu, je souhaitais aussi rentrer dans la communauté du sciML car florissant et surtout axé pratique
l'idée c'était de donner des insights théoriques sur comment choisir en pratique en 1 shot la taille de notre réseau

1)
notre idée pour attaquer le loss landscape c'est que l'on conjecture que le NTK décrit très bien le loss landscape jusquà un point
et on peut essayer de recoller les analyses locales pour en faire une globale



les resultas de terjek maintenant


Concrètement ce résultat dit que si vous avez beaucoup de points, vous devez constraindre l'espace modèle en augmentant s, car en profitant de l'information
sur la smoothness dans les données, ne pas l'utiliser rend le problème plus difficile, vous allez avoir pleins de directions de l'espace
des paramètres où la loss ne bouge pas, dans ces directions les fonctions candidates n'utilisent pas l'information de la données

avec un grand s, vous contrevenez à cela et régularisez le problème, à raison
et en terme d'approximation justement d'autres idées sont à venir (notamment résultats de doumeche) car augmenter s à un impact dessus
et justement haizhao yang mon prof a été celui qui a obtenu les derniers résultats avec des tights bounds pour des bornes de geenralization et approx
pour FCNN en sobolev training
donc on est en train d'assemnbler le puzzle complet


pour moi ça constitue mon premier résultat majeur qui est à la fois très intuitif, rigoureux, à haut potentiel de généralisation pour n'importe quel schéma de données
et distributions d'entrées et je suis en cours de rédactions des preuves (il faut controler les produits scalaires d'harmoniques sphériques by sampling et faire des proba high dimensionnal etc ..) puis utiliser des bornes type bernstein et d'eigenvalues, hanson wright inequalities for random tensors, vershynyn etc ..


je vais poursuivre ça cette année en collaborant si possible avec des profs du labo de proba d'orsay

 Pour la construction de mes papiers, on va donc donner en plus de rafiner les théorèmes des expériences numériques qui comparent
c'est à dire mettre un résultat d'à quoi ressemble le loss landscape, et à travers des plots et des training en analysant 
au cours du training la courbure, voir si on a bien prédit la loss finale, et quels sont les mauvaises distributions
donner des insights pratiques sur quelle distribution ou quel dataset nos résulats peuvent déjà s'appliquer


Donc là on a disentangle le problème en mettant la dimension d dans le sobolev, ainsi que la régularité de f aussi
la deuxieme chose importante c'est donc la taille de votre model space et comment trouver le bon model space
c'est à dire dans la classe NN comment trouver la bonne architecture, bonne activation pour ça
on est donc en train d'essayer de trouver des techniques de préconditionnement du NTK


ici on est vraiment dans l'approximation de fonction, on veut construire des PINNs qui approximent bien des fonctions 
solutions d'edp


2)

Mes superviseurs ont bossé sur le sujet, et ont essayé de trouver des candidats empiriquement qui en ayant des thm d'approx et generalization quasiment identiques,
permettent de préconditionner l'optimisation afin d'avoir un impact pratique rapide et théorique long terme, et parmi les candidats on a les MMNN et transformers
 

Pour les MMNN, je rappelle que la construction est comme ça, et l'idée est d'avoir une matrice a milieu low rank, 
ne train une partie parce que les stiefel riemannian descent c'est beaucoup trop dur, et ne train que la 1er partie de la matrice
pour interpreter ce que l'on fait comme une concatenation de random basis functions et en 1 diminuer d'un ordre de grandeur le nombre de paramètres

L'interpretation frequentielle est que les random basis, avec des coefficients a et -a, et avec le biais selectionnent des portions d'intervalles
et approximent linéairement dessus, donc on apprend A la pente locale, ce qui permet avec la meme activation d'apprendre sur plein d'intervalles differents
differentes frequences. L'insight majeur que l'on donne c'est qu el'on est en train d'apprendre la transformée en ondelette
et les cas ou les mmnn performent le mieux sont precisement quand la transformée en ondelette est sparse avec quelques spikes

ça c'est pour l'approximation generalisation


les expériences numériques nous ont montré que déjà les MMNN permettent de trouver des minima globaux assez rapidement et avec une précision maximale
c'est à dire super convergence, cela indique que le loss landscape en plus est propice à ça ! 

Methodologiquement, on trace des loss landscape et on compare les MMNN à leurs equivalents FCNN, pour différents settings
et on voit que pas mal d'initialization font apparaitre un loss landscape montagneux et pas du tout non propice

et ça, en terme local, ça doit se voire dans le NTK, et en poussant loin les analyse NTK on doit pouvoir donner une réponse.

Donc on fait d'abord des experiences numeriques du NTK pour des réseaux MMNNs, on calcule des matrices de gram, spectre
on vérifie la cohérence du scaling des valeurs prores en fonction du nombre de paramètres par exemple pour 1 couche

et surtout le but c'est d'obtenir un élement de comparaison avec le FCNN, c'est à dire à 2 couches, quand un low rank apparait
comparer un 2 layer mmnn à un 3 layer FCNN c'est à dire 3 hidden layers FCNN là où le low rank apparait


le fait que le nb de poids soit linéaire en width fait que se placer dans le NTK regime n'est pas couteux du tout ! alors que c'est en n^4 pour FCNN, ici c'est en n²
avec n la taille du dataset donc totalement pratique
le but c'est de donner
donc on s'attaque au NTK



et avec la caractérisaiton du champ, on en déduit le NTk de deux maniere:

soit naivement vous calculez le NTK et vous faire tendre à la limite on identifie avec des TCL et c'est terminé
c'est rapide mais on n'interprete rien , j'ai fait ça la preuve est en appendix,
mais cette technique c'est surtout pratique pour obtenir une relation de recurrence, que j'ai donc aussi obtenu
et qui montre la différence entre un FCNN et un MMNN



la 2eme methode c'est de trouver la loi limite du random field, et d'en déduire le NTK par un théorème d'adler, et c'est ce que j'ai fait







d'abord on essaye d'avoir un théorème de structure de la loi du random field en sortie, gaussien ou pas, en réalité c'est un deep gaussien, c'est à dire
une concaténation (supportable théoriquement et numeriquemetn par des intégrales et operateurs de convolutions, analyse point fixe possible )






La suite à donner pour la consruction de mes papiers c'est de lancer un training justement en comparant les prédictions théoriques et la pratique
que ce soit pour les loss finale mais aussi pour les courbures, le comportement du training et sa prédiction apr le NTK
et surtout la suite c'est de recoller les analyses globales, 

la relation de recurrence nous permet de passer à une depth quelconque, de faire une analyse du scaling en collab avec terjek
faire une analyse en proba de lyapunov pour le produit consecutif, et 
la finalité très technique d'ailleurs on peut meme de cette maniere retrouver le rkhs en faisant un developpement de puiseux (taylor non entier)
au voisinage de 1, retrouver le RKHS du random basis model avec papier de francis bach/bietti, puis vu que les MMNN sont surtout utilisés en PINNs on peut appliquer mes résultats précedents
ce developpement de taylor nous donne aussi, quand poussé à 3-4 termes, une description precise du spectre et de son scaling
en fonction de N et L

et enfin aboutir à un compromis entre RKHS approx/gene et optimization 
le but c'est que dans 1 an ou 2 on sache ce que l'on mette quand on ouvre une librairie de sciml, de sorte
à ce que ne soit plus vous qui, avec trial and error sur depth width pour trouver la meilleure loss, l'heuristique soit beaucoup plus directe
et les parametres par defauts/recommandations pointillleuses
tout le monde et surtout le public n'a pas accès à 10k gpu




3)
Afin de répondre à la question deeper or wider, on essaye donc de calculer le ntk pour un reseau fini
et cette théorie est connue et travaillée de seulement 10 personnes, beaucoup de physiciens

l'iée est que un réseau fini = un NTK qui bouge, donc pour caractériser un NTK fini
pour des valeurs de N quelconque on essaye de voir quel ODE le NTK vérifie, on dérive par rapport au temps et 
et on voit que l'on peut obtenir très facilement une structure hiérarchique infinie de NTK
que l'on appelle la NTH.

Cette NTH fait apparaitre des tenseurs, issus des dérivées par rapport aux poids de la loss, 
des dérivées du NTK à tout ordre.

et dans ces tenseurs, il se trouve que l'on somme plein de termes et de produits de poids, activations et
préactivations.

généralement, le tenseur d'ordre n fait apparaitre des correlations d'ordre 2n, entre des variables gaussiennes
(gaussiennes par TCL ou gaussienne par poids).


donc les physiciens quand ils ont vu ça, notamment ethan dyer et guy gur ari, on tout de suite reconnu
que les correlations etaient très similaires à celle que l'on obtient quand on calcule
des integrales sur des champs gaussiens avec termes pertubatifs, que l'on calcule 
en faisant une série et comptant les corrélations avec diagrammes de feynman
c'est le théorème de wick que l'on applique pour tous les moments (qui en fait est une formule
combinatoire pour des décompositions d'hermite multivariées)


avec ceci, ils ont conjecturé tres facilement puis prouvé que les corrélations d'ordre 2n
décroissaient avec scaling 1/m^(n/2) avec m la width,


et c'est de là qu'est arrivé la décomposition asymptotique du NTK en puissance de 1/M


donc quand on essaye de decrire comment le NTK bouge en fonction du temps
on décrit en même temps le NTK fini


j'étudie donc les tenseurs d'ordre 3 et 4, je dérive des formules avec un système de réécriture
introduit par les chercheurs (qui vient de chain rule), je suis careful sur mon hypothèse que relu est problématique en 0
donc je supporte ça par un théorème obscure (lee)
et je trouve ddonc une formulation simple pour 3 mais encore trop dure pour 4
(logique diagramme croisse exponentiellement)



donc je fais des experiences numériques pour essayer de voir comment tout ça se comporte avec la profondeur
dans l'idée d'appliquer l'inégalité de weyl et directment borner une smallest eigenvalue

donc je dois essayer une approche numérique ET théorique

quand j'essaye mon approche théorique je me rends compte qu'avec le résultat en décomposant sur les valeurs propres du NTK, on trouve une formule qui ressemble étrangement à une moyenne
encore plus quand on sait que toutes les eigenvalues sont dans un bulk empiriquement hormis 1

ce qui me permet d'écrire la moyenne sur O3 et O4, avec l'idée que mes moyennes dans un bulk
proviennent d'une moyenne sur des valeurs propres suivant une certaines loi limite dans le bulk
que j'ai conjecturé comme étant une MP  à cause de la structure de covariance et ça s'est avéré vrai par la suite (voir papier benigni)
je me rappelle aussi que le NTK étant invariant par rotation, je devrais voir apparaitre une matrice
diagonalisaante orhtogonale et si possible uniforme car très classique en rmt. 

donc je formule une hypothèse la plus scale invariante et dscriminante possible, celle sur le wigner surmrise
des ratios d'eigenvalues spacing et badaboum j'obtiens ça, mon hypothèse s'est avérée vraie !!




en terme computationnel on a une complexité quartique avec les tenseurs mais maintenant on transforme 
ça en une moyenne beaucoup plus facile à calculer numériquement et forme close


donc voilà je suis en train de discuter avec tous les chercheurs actuels pour coller les bouts ensemble
et enfin aboutir à un scaling sur la depth. le but est de recalculer chaque terme de correlation avec 
cette moyenne et transformer un monte carlo en une forme close, et sommer tout ça avec controle en finite width
et dataset fini


le decouverte sur la repulsion des eigenvalue est encore plus fort, car on a la structure locale en plus de globale MP
précédemment d'autres personnes avaient vu à travers la hessienne une fois que le réseau était entrainé notamment mais là c'est un nouvel angle prometteur car la hessienne on sait pas la calculer mais le NTK on peut completement ! avant et après initialisation







- on présente les approches finales physiques et ce que les gens pensent
- là les suites que je donne au moins sur les aspects discutés, puis les aspects omis & futurs, et la réelle fin et collaborations

- la fin : leçons tirées, quelle suite donner dans le futur, mon avis sur après tout ça, 







une fois ceci fiat, on n'est UNIQUEMENT à l'initialization, 
nous on veut recoller les analyses ça requiert la hiérarchie
La communauté de l'optimization a délaissé le NTK pour se focus sur des algorithmes pratiques depuis 2022
et avec les llm c'était terminé il n'y a plus personne dessus, environ 30 auteurs et coauteurs differents sur
les 3 dernieres années

parce que le NTK avait l'air trop théorique, trop focus sur l'intialisation
et effectivement on ne donne pas d'insight global, seulement local

donc pour donner une direction pratique qui réconcillie la théorie et essayer de redonner ses lettres de noblsee
on pense, moi et mon professeur, en ayant discuté avec la commu du NTK que j'ai contacté,
que l'on peut coller les analyses locales et en faire une globale.

le NTK nous donne la cuvette locale (et la courbure locale à cause de ses liens avec la hessienne)
et prendre t infini en finite width nous donne un point (un peu comme une methode de newton du 2nd ordre )
et on itère comme ça, on obtient une suite qui on l'espère converge vers un minimum global
(ou non et dans ce cas là on caractérise la complexité de l'optimization hautement non convexe d'un NN)













approches physiques : la rmt rely beaucoup sur la combinatoire et les dérivés de wick
donc en fait c'est surtout 2 approches, 1 field theory qui fait une analogie complète en terme de corrélations, systèmes en intéraction etc ..

et la phystat, l'approche d'un système complet à grande échelle à l'équilibre avec les analogies verres etc ..

et le pont entre les 2 est la RMT, et les 3 matrices que sont NTK, Gauss newton et hessienne, et le noyau NTK
très précisément le spectre
là présenter le papier de lucas benigni, la caractérisation du spectre en régime infinite width en tant que matrice de covariance, donc un dérivée de marchenko pastur
qui dépend très fortement de l'activation et de sa non linéarité, spécialement du scaling de ses coefficients d'hermite (c'est à dire du dev de taylor de la fonction rho)
On le savait déjà avec 'spectrum of kernel random matriecs' mais là on a la distribution empirique du spectre en grande dimension et infinite width
ça reste encore très simplifié comme approche, c'est plus de 5 ans de travail pour LB pour arriver là

et le coeur de la preuve c'est le controle des correlations calculées à travers des diagrammes de feynmann pour méthode de moments pour controler les inégalités et scalings, et transformée de stieltjes pour caractériser les distributions



en ce moment je collabore avec mg&mis pour justement, à l'issue de leur papier de fin aout, calculer les corrélations cette fois ci 
dans les finite width avec des diagrammes de feynman qu'ils ont introduit, aller au delà des expériences numériques (en lancer des grosses)
en trouvant un scaling cohérent pour la profondeur


après ce gros travail on aura en place la plupart des lemmes pour donner une réponse finale, pour des réseaux finis, d'à quoi ressemble le loss landscape
et de répondre si oui ou non nos algorithmes actuels ont implicitement et par hasard permis de contourner ces difficultés


la suite à donner pour statistiques locales (et non globales), pour la depth, ensuite donner une analyse pratique d'Adam & Muon
et finalement comprendre le loss landscape

je souhaite vraiment rester ancré dans la pratique aussi, ne pas m'enfermer dans mon coin, c'est pour ça que j'ai choisi comme point d'entrée le sciml
et comme point de sortie les transformers. 

Pour moi ce n'est que le début, on a des résultats très intéressants et je n'ai pas tout présenté pour rester concis, nos communications et publications le seront davantage
pour profiter à la communauté open source







Conclusion : 
honentement j'ai rarement été aussi heureux depuis que je travaille dessus, 

Il y a une partie de mon travail qui est très pratique, notamment quand j'explique quel schéma de point garanti une compréhension parfaite du training
je donne une borne pratique pour des tailles de réseaux raisonnables qui sont utilisables dès maitenant et que je suis en train de raffiner
notamment en donnant dans les prochaines semaines une preuve de convergence avec la finite width.

Je mets en valeur certains aspects théoriques, notamment le GOE a un impact, il dit que les eigenvalues sont espacés, ce qui signifie
que le NTK spectrum est très bien approximé par sa distribution limite


- lecons tirées : trouver le problème le plus trivial à résoudre en premier
En terme de méthodologie de travail j'ai beaucoup appris sur moi aussi, j'avais un rythme pas comme en prépa, car ce qui compte c'est la créativité et la
représentation mentale du problème que je me faist



j'ai documenté le plus possible tout ce que j'ai fait, sur github, sur youtube quand je code ou brainstorm mes idées, sur papiers en scannant toutes mes feuilels (que je suis en train d'upload) parce que j'aime bien partager aux gens ce que je fais; je publierai et mettrai les vid et tt en public et communiquerai dessus après le jury 3A 

je suis ressorrti avec des collaborations avec plus de 10 personnes que j'ai contacté actuellement avec qui j'échange sur le sujet, et davantage de questions que de réponses
on s'échange des papiers et on discute de nos idées de preuves quand on sèche