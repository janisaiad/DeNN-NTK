en fait pour repondre a la question on a plusieurs chemisn

deja on peut regarder la correction finie de ntk, en fiat c'est ce qui permet de calculer des quantités liées au ntk sans devoir le calculer explicitement
c'est juste ecrire les sommes matricielles d'une maniere qui nous arrange en faisant apparaitre les representations en 1/M
au fond c'est ça que ça veut dire , et avec ça on peut deja etudier comment le ntk scale avec L m N et d

ca nous donne l'etude de o3 et O4 mais ils apparaissent sous une ceratine forme
en fait on a des formules pour eux qui sont tres longues et qui sont ponderees par le spectre du ntk mais du ntk infinite width
qui est tres facile a calculer
de la on peut etudier directment les noyaux et essayer de trouver les scalings associes, combines aux sacling des valeurs propres
 et on a des insights en fonction de L
 car on a une somme d'une certaine quantité de kernel (qu'on pourrait d'ailleurs etudier separement)
 àopur les etudier separement ça peut etre pratique pour o3 mais ps pour o4,
 car il y en a trop
 en fait pour o4 on peut calculer en fonction de la hessienne du ntk, et le ntk est implémenté de maniere autograd, 
 donc on peut le faire avec un autograd pour fair eplein de calculs, la procedure est la suivante, 
 et ainsi ca permet de voir les caling des valeurs,
 mais encore mieux on peut faire quelque chose, c'est que le spectre peut de maniere statistique
 avoir certaine particularités notamment les eigenvectors ont une ceratine distribution, cela vient de l'aspect rmt,
 la distribution peut etre predie par certains trucs rmt mais c'est assez dur car il n'y a pas de gaussianité, il faudrait pour cela non pas etudier le spectre mais 
 vraiment les eigenvector, et pour cela regarder sous quel groupe de symetrie (representation theory) le K3 est invariant 
 pour recuperer des invariants de symetries dans la distribution des eigenvectors qui la caractériseraient (par exemple la norme pour GUE)

 ça c'est pour l'aspect preuve, mais les eigenvectors ont aussi une structure de dependance non evidente dans o4 qui est
 à expliciter (JOBBBBBBBBB)
 une fois cela fait on peut obtenir des vraies insights sur comment le noyau agit, parce qu'en plus o4 est très proche à un ordre de grandeur en plus
 de sa version infinite width, donc on peut faire des estimations pour un M assez grand mais pas trop (et qui limitera les calculs)

 on peut aussi se mettre à calculer leur moyenne si de maniere statistique ça passe, et qu'on a une indépendance (pas sure) entre eigenvector ntk et les termes
 internes de o4 (ce qui est le ca en fait car les eigenvectors infinite width ne dependant pas !!) donc on peut les sorties
 bref le calcul de la moyenne mathamtique peut etre assez facilité sous la conjecture qqqch

cette correction permet en fait surtout d'appliquer l'ingéalité de weul, qui est sharp dans le pire cas ou les eigenvectors sont paralleles entre les 2 matricees
pour la smallest eigenvalue

et l'interpretation a en faire est que sortir du regime ntk a un cout qui peut etre compensé par la la profondeur ou baisser la quantité de données
(augmenter ou baisser la profondeur c'est ça la question majeure) et baisser la quantité de données, 

ça c'est l'aspect correction. en fait c'est vraiment depenendant des corrrelations enter eigenvectors et de comment les calculs se font avec la moyenne
d'ailleurs l'influence de K4 est importante car c'est l'influence hessienne du NTK, c'est opur ça qu'elle compte le plus 
alors que l'influence linéaire du ntk est plus faible

attention il faut faire attention à la differentiabilité pour relu, car c'est un aspect important, meme s'il y a un aspect mesure nulle qui rend la chose facile
pour une parametrisation qui rend la chose avantageuse (par exemple mu p ou NTK) avec des poids très petit mais qui bougent beaucoup 
on peut aussi voir cet aspect mesure nulle dans le NTK narrow





pour les autres pistes qui permettent de repondre à cette question, et de pourquoi elle est importante, il ya au ssi l'aspect hessien,
en fait le NTK et la hessienne ont un spectre très proche, au moins pour certaines eigenvalues (les D premieres), ça tient avec une conjecture, ou beaucoup
de calculs anciens, et c'est pourquoi en fait les 2 points de vues sont les memes, 

ça tient aussi car le NTK c'est la matrice de gauss newton, et que les termes non diagonaux sont assez proches des termes diagonaux (seleznova)

d'ailleurs seleznova dit quelque chose d'interessant, c'est que la variance du NTK explose,avec le ratio
et c'est un resultat que l'on doit retrouver quand on regarde la finite width, la variance quelque part doit exploser si on n'est pas à l'eoc

ces aspects diagonaux et non diagonaux apparaissent quand on etudier les random kernel matrices (el karoui)
meme a l'eoc on a une dispersion qui eclate pour des reseaux vraiment rectangle

le truc c'est que meme pour un aspect optimisation de taille c'est pas optimla, et pour un aspect concentration c'est vraiment sous optimal
comme le papier de terjek le dit, on peut avoir une concentration du NTK vraiment bien selon la forme du reseau


on peut meme s'amuser à voir que vaut le lambda dans ce cas de réseau parabole, car la somme va converger donc on ne se retrouve plus avec L/M mais une constante
autre qui tend vers O quand m augmente je crois


dans cet aspect parabolique il y a aussi un aspect quel layer je fais croitre à l'infini car apparement on peut seulement se contenter d'une layer qui diverge
pour pouvoir appliquer la theorie du NTK

en tout cas si on se place dans un cadre parabolique on peut bien etudier la theorie du ntk, et au moins ses perturbations finies
(on peut essayer de faire des experiences sur ces reseaux paraboliques)
(d'ailleurs le ntk de seleznova se calcule super facilement, je vais reprendr eleur implementation)
d'ailleurs donner un sens de largeur moyenne ne marche pas car la somme est dominée par le dernier terme

en fait le prix à payer en terme de neurone devient quadratique en alpha mais le coeff lambda est en 1/alpha
donc il y a un compromis à voir pour trouver le meilleur alpha quand on a un certain nombre de donneés, (parce qu'avec le budget ça passe)
alors qu'avec une croissance linéaire ça diverge et donc c foutu, le carré est important car il minimise VRAIMENT la variance
(voir dans la preuve pourquoi c'est vraiment le cas)

au moins on peut s'assurer que l'on peut rester dans le cadre NTK et calculer des moyennes AVEC DU SENS, car sinon ça n'en a pas !!
puisque la dispersion et le post training va trop bouger et ça ne fonctionnera plus
à mettre en perspective avec le papier NTH, parce qu'avec une architecture changeante ça change les plans, on peut toujours
essayer de calculer une déviation mais là ce sera pas en 1/M masi en 1/Q avec Q la moyenne geometrique des m_l (je crois, selon les papiers de physique theorique), scaling moyen

mais le scaling moyen lui est un peu complexe car la somme des log nous donne un nlog(n) donc scaling en 1/m*L² moyen je crois (a investiguer)



le  but de toute cette discussion c'est d'etre vraiment rigoureux sur quel theoreme on peut appliquer dans quel cadre, pour avoir des estimations
probabilistes qui sont bien (variance diminuée), avec une bonne confiance en les corrections finies, la theorie du NTK est bien car elle a de gros liens
avec la hessienne, et que la hessienne c'est pas forcement un truc pratique des que notre reseau devient complexe (block hessian pour transformers,
d'aillleurs montrer une gueule de hessienne de NN)

l'avantage c'est surtout qu'à l'initialisation et APRES le training on a des garanties, c'est ça qui est interessant avec le NTK (et qui ne l'est pas quand
on etudie des optimizationo bounds sous l'angle SGD etc .. car là on peut se faire avoir par le optimization landscape ou les minima globaux sont locaux)
(d'ailleurs c'est la grand question du deep learning, les minima locaux sont globaux ?) peut etre sous l'angle morse theory

peut etre que ducoup sous cette hypothese c'est pas le NTK qu'il faut voir

c'est pour ça que recentrer la discussion du NTK sur l'applicabilité, les hypothèses et le cadre high dimensionnal, high compute, high size est bien (sinon
ça devient trop difficile )

à noter aussi les constantes de variance du ntk de terjek

en fait la perspective du NTK quand m est grand et avec o3 o4 c'est de répondre pendant le training
de sorte à pouvoir obtenir une optimization bound en fonction de L et M

il y a aussi un aspect important, ce n'est pas la depth qui change la structure profonde du spectre, elle l'élargi de manière linéaire
mais le spectre garde sa forme, on ne pourra jamais contrer le conditionnement ça c'est sur c'est écrit dans les équations

d'ailleurs le 1er vecteur propre est celui des normes, donc c'est celui qui dit qu'en cas d'homogénéité, c'est d'abord la norme qui compte
(puisqu'elle est propagée) et donc d'abord on apprend quel scaling il faut pour chaque vecteur (en fonction de sa norme donc)
car c'est la "scaling appropriée de premier instance qui permettra ensutie de bien approximer la fonction f (impossible d'apprendre un relu à l'approximer
sans d'abord apprendre la norme), ça c'est ce que le NTK nous dit c'est qu'agumenter la profondeur ça marche, et qui scale bien en fonction de L
donc le pouvoir séparateur des normes (je suis pas sur de ca) augmente avec L, mais l'autre pouvoir aussi (faire
un dessin du pouvoir separateur des normes sur une sphere, et pourquoi ça augmente, parce qu'on rajoute des chemins ?)



et après cela fait c'est la partie autre du spectre qui importe et c'est celle là qui nous intéressent, on veut apprendre les patterns
et là le ntk ne le décrit pas bien en fait, car le spectre ne s'approche vraiment pas du spectre qu'il nous faut, avec une separation
(à verifier avec MP, refaire calculs) des eigenvalue quasi identiques

donc dans les nouveaux aspects mathémtiques à introduire pour analyser ça on a le narrow ntk, que l'on pourrait développer dans un aspect purement creatif


ce qu'il faut faire :
traiter les correlations eigenvectors
verifier le MP pour les scalings inter valeurs propres

verifier les scalings dans o3 o4, puis verifier les scalings internes
implementer o4 en 
montrer une gueule de hessienne en comparaison avec le NTK

faire les liens terjek variance seleznova
release le code en mode n'importe qui peut telecharger et run, une fois les optimization faites

faire des diapos comme la soutenance, rapport comme la soutenance (car on est à la moitié, enfin commencer pour la semaine pro avancer, voir au feeling
si lex experiences disent uqelque chose et sinon dire que la semaine prochaine je faerfais aca)

pas oublier que les correlations qu'on calcule c pas grave à conjugaison pres (dans la base interne)

en deuxieme instance : 

tester le NTK en dsrn
en fait otutes les experimentations on peut les lancer sur le tore et sur espace gaussien
il faudrait aussi traiter shiijun et rank mmfn ffnn, et aussi pour haizhao en exposant fractionnaire l'approximation
traiter aussi la dimensionnalité pour le sobolev training
traiter les experiences des autres codes pour voir ce qu'ils donnent, et traiter le code en jax aussi
surtout le narrow en creatif sur une vm
ecrire la variable aleatoire du NTK en faisant la somme de chemin à la physicienne (ising ?)



script of what said : 



