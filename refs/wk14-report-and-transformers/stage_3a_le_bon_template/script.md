Bonjpur je suis très content de vous avoir maintenant pour vous présenter mon travail durant les
4 derniers mois. Vraiment très content car vous êtes physicienne de formation et que la réalité mathématique derrière nos calculs en deep learning est visible
à travers des analogies physiques


Avec mon superviseur, on souhaitait répondre de manière long terme à la question de l'optimization
landscape pour les réseaux de neurones, et plus spécialement pouvoir donner une réponse one shot 
au design optimal et préconditionnement à travers l'architecture. Comprendre pourquoi c'est difficile
et que faire pour un budget donné de paramètres. J'ai répondu à cette question dans mon rapport donc là
je vais surtout me focus sur la démarche et le pourquoi, l'intuition que l'on a eu à travers les discussions et collaborations informelles.

Au delà d'avoir 100k gpu, je souhaitais aussi rentrer dans la communauté du sciML car florissant et surtout axé pratique
l'idée c'était de donner des insights théoriques sur comment choisir en pratique en 1 shot la taille de notre réseau

1)
notre idée pour attaquer le loss landscape c'est que l'on conjecture que le NTK décrit très bien le loss landscape jusquà un certain moment, et on recommence à partir de ce moment, on retrack la dynamique gradient avec le NTK à partir de ce nouveau point et peut essayer de recoller les analyses locales pour en faire une analyse globale


Concrètement ce résultat dit que si vous avez beaucoup de points, vous devez constraindre l'espace modèle en augmentant s, car en profitant de l'information
sur la smoothness dans les données, ne pas l'utiliser rend le problème plus mal conditionné, vous allez avoir pleins de directions de l'espace
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

l'idée est que un réseau fini = un NTK qui bouge, donc pour caractériser un NTK fini
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