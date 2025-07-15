en fait pour repondre a la question on a plusieurs chemisn

deja on peut regarder la correction finie de ntk, en fiat c'est ce qui permet de calculer des quantités liées au ntk sans devoir le calculer explicitement, cependant ça requiert de calculer K3 et K4, de manière  automatique ou avec les formules,
avec les formules c'est assez long comme calcul mais ça peut se faire, ON DEVRAIT BENCHMARKER LE TEMPS PRIS


au fond c'est juste ecrire les sommes matricielles d'une maniere qui nous arrange pour que K3 et K4 s'implémentent facilement en JAX avec FLAX, en faisant apparaitre les representations en 1/M


ca nous donne l'etude de o3 et O4 mais ils apparaissent sous une ceratine forme
en fait on a des formules pour eux qui sont tres longues et qui sont ponderees par le spectre du ntk mais du ntk infinite width !
dont le spectre est tres facile a calculer et on a remarqué qu'il y avait une statistique GOE assez marquante ! ce qui traduit une
gaussiannité/uniformité quelque part

un fait important est que ces scalings pour K3/K4 font apparaître ces noyaux à l'ordre infini ! donc on peut
calculer la limite pour M grand et prendre cette limite pour etre sur (nulle pour K3, et avec une déviation de m^1.5 pour K4 )

de la on peut etudier directement les noyaux eux meme (donc leur limite ou déviation limité) et essayer de trouver les scalings associes avec L N D, combinés aux sacling des valeurs propres
- on a déja des insights en fonction de L avec le nombre de termes, masi il faudrait évaluer chacun d'eux avec un exposant de lyapunov, on étudie ces noyaux séparément 
- ppur les etudier separement ça peut etre pratique pour o3 mais ps pour o4, car il y en a trop





AFAIRE : en revanche pour o4 on peut le calculer en fonction de la hessienne du ntk, et le ntk est implémenté de maniere autograd en flax/jax
donc on peut le faire avec un autograd pour faire plein de calculs, la procedure est la suivante, et ça n'a je pense pas été fait 
car j'ai cherché partout je n'ai rien trouvé
ça permettra d'avoir un code beaucoup plus efficient et ainsi ca permet de voir le scaling pour N et L

PREUVE : mais encore mieux on peut faire quelque chose, c'est que le spectre peut de maniere statistique avoir certaine particularités notamment les eigenvectors ont une ceratine distribution, cela vient de l'aspect rmt, 
la distribution peut etre predie par certains trucs rmt mais c'est assez dur car il n'y a pas de gaussianité, il faudrait pour cela non pas etudier le spectre mais  vraiment les eigenvector, et pour cela regarder sous quel groupe de symetrie (representation theory) sous lequel K3 est invariant, pour recuperer des invariants de symetries dans la distribution des eigenvectors qui la caractériseraient (par exemple symétrie de la norme pour GOE)


ça c'est pour l'aspect preuve, mais les eigenvectors ont aussi une structure de dependance non evidente dans o4 qui est
à expliciter (JOBBBBBBBBB), car on a une forme multilinéaire sur lequel on bosse et qui peut etre liée à o4 de manière non triviale
une fois cela fait on peut obtenir des vraies insights sur comment le noyau agit, parce qu'en plus o4 est très proche à un ordre de grandeur en plus
de sa version infinite width, donc on peut faire des estimations pour un M assez grand mais pas trop (et qui limitera les calculs)

on peut aussi se mettre à calculer leur moyenne si de maniere statistique ça passe, et qu'on a une indépendance (pas sure) entre eigenvector ntk et les termes internes de o4 (ce qui est le ca en fait car les eigenvectors infinite width ne dependant pas !!) donc on peut les sorties bref le calcul de la moyenne mathematiquues peut etre assez facilité sous la conjecture de gaussiannité

cette correction permet en fait surtout d'appliquer l'ingéalité de weyl, qui est sharp dans le pire cas ou les eigenvectors sont paralleles entre les 2 matrices pour la smallest eigenvalue, chose qui n'est pas forcément évident

et l'interpretation a en faire est que sortir du regime ntk a un cout qui peut etre compensé par la la profondeur ou baisser la quantité de données (augmenter ou baisser la profondeur c'est ça la question majeure) et baisser la quantité de données)



ça c'est l'aspect correction. en fait c'est vraiment depenendant des corrrelations enter eigenvectors et de comment les calculs se font avec la moyenne d'ailleurs l'influence de K4 est importante car c'est l'influence hessienne du NTK, c'est pour ça qu'elle compte le plus alors que l'influence linéaire du ntk est plus faible, en tout cas c'est ce qu'on peut voir 
sur quand on sort les termes et le scaling de Terjek


attention il faut faire attention à la differentiabilité pour relu, car c'est un aspect important, meme s'il y a un aspect mesure nulle qui rend la chose facile, pour une parametrisation qui rend la chose avantageuse (par exemple mu p ou NTK) avec des poids très petit mais qui bougent beaucoup, car la dérivation symbolique des noyaux reste vraie sous réserve de la différentiabilité du NTK, 

on peut aussi voir cet aspect mesure nulle dans le NTK narrow, c'est un papier qui est vraiment riche pour le futur je pense
seul bémol c'est que le NTK est vraiment simple et que la paramétrisation avec l'hypothèse d'être positif rend constant les feature maps en fonction de L


pour les autres pistes qui permettent de repondre à cette question, et de pourquoi elle est importante, il ya au ssi l'aspect hessien, en fait le NTK et la hessienne ont un spectre très proche, au moins pour certaines eigenvalues (les D premieres), ça tient avec une conjecture sur les feynman diagrams sous la conjecture de calculs de moments de Dyer, ou alors beaucoup de calculs anciens de Jacot, et c'est pourquoi en fait les 2 points de vues sont les memes : la hessienne totale c'est le NTK*hessiene loss +gradloss*hessf

et on a des résultats (appearance) qui disent que pour un GP, la hessienne a des statistiques gaussiennes locales mais non globales
(Empirically we actually find the even stronger bound Eθ
[
λ(H)
i − λ(Θ)
i
]
= O(n−1) for the top Din eigenvalue differences
and O(n−1/2) for the remaining eigenvalues in the one hidden layer case. We can gain insight into this improved scaling
through the perspective of degenerate eigenvalue perturbation theory [32], but this is outside the scope of the current
presentation.
)

c'est super intéressant aussi car le NTK c'est la matrice de gauss newton, et que les termes non diagonaux sont assez proches des termes diagonaux (seleznova) (en tout cas dans le régime grand dimensions d'entrée, on a des théorèmes de selez et avec curse of dim)

d'ailleurs seleznova dit quelque chose d'interessant, c'est que la variance du NTK explose, avec le ratio
et c'est un resultat que l'on doit retrouver quand on regarde la finite width, la variance quelque part doit exploser si on n'est pas à l'eoc peut etre (pour K3 et K4, on doit pouvoir le retrouver dans les formules et aussi numériquement)


ces aspects diagonaux et non diagonaux apparaissent quand on etudier les random kernel matrices (el karoui)
meme a l'eoc on a une dispersion qui eclate pour des reseaux vraiment rectangle

le truc c'est que meme pour un aspect optimisation de taille c'est pas optimla, et pour un aspect concentration c'est vraiment sous optimal comme le papier de terjek le dit, on peut avoir une concentration du NTK vraiment bien selon la forme du reseau, avec quelque chose paraboliguqe


on peut meme s'amuser à voir que vaut le lambda dans ce cas de réseau parabole, car la somme va converger donc on ne se retrouve plus avec L/M mais une constante/m avec m hyperparam, et on augmente un peu M, ça tend vers 0


dans cet aspect parabolique il y a aussi un aspect quel layer je fais croitre à l'infini car apparement on peut seulement se contenter d'une layer qui diverge pour pouvoir appliquer la theorie du NTK, et c'est super, puisqu'on a l'hypothèse de pas
avoir de bottleneck (à formaliser dans Tight bounds)


en tout cas j'ai une idée c'est que si on se place dans un cadre parabolique on peut bien etudier la theorie du ntk, et au moins ses perturbations finies (on peut essayer de faire des experiences sur ces reseaux paraboliques)
(d'ailleurs le ntk de seleznova se calcule super facilement, je vais reprendr eleur implementati)
d'ailleurs donner un sens de largeur moyenne ne marche pas car la somme est dominée par le dernier terme, c'est super important


en fait le prix à payer en terme de neurone devient quadratique en alpha mais le coeff lambda est en 1/alpha
donc il y a un compromis à voir pour trouver le meilleur alpha quand on a un certain nombre de donneés, (parce qu'avec le budget ça passe), 
alors qu'avec une croissance linéaire ça diverge et donc c foutu, le carré est important car il minimise VRAIMENT la variance
(voir dans la preuve pourquoi c'est vraiment le cas)

au moins on peut s'assurer que l'on peut rester dans le cadre NTK et calculer des moyennes sur des features maps AVEC DU SENS, car sinon ça n'en a pas !!
puisque la dispersion et le post training va trop bouger et ça ne fonctionnera plus 
à mettre en perspective avec le papier NTH, parce qu'avec une architecture changeante en fonction de L ça change les plans, on peut toujours essayer de calculer une déviation mais là ce sera pas en 1/M masi en 1/Q avec Q la moyenne geometrique des m_l (je crois, selon les papiers de physique theorique), scaling moyen

mais le scaling moyen lui est un peu complexe car la somme des log nous donne un nlog(n) donc scaling en 1/m*L² moyen je crois (a investiguer)



le but de toute cette discussion c'est d'etre vraiment rigoureux sur quel theoreme on peut appliquer dans quel cadre, pour avoir des estimations probabilistes qui sont bien (variance diminuée), avec une bonne confiance en les corrections finies, la theorie du NTK est bien car elle a de gros liens avec la hessienne, et que la hessienne c'est pas forcement un truc pratique des que notre reseau devient complexe (block hessian pour transformers, d'aillleurs montrer une gueule de hessienne de NN)

l'avantage c'est surtout qu'à l'initialisation et APRES le training on a des garanties, c'est ça qui est interessant avec le NTK (et qui ne l'est pas quand on etudie des optimizationo bounds sous l'angle SGD etc .. car là on peut se faire avoir par le optimization landscape ou les minima globaux sont locaux)
(d'ailleurs c'est la grand question du deep learning, les minima locaux sont globaux ?) peut etre sous l'angle morse theory

peut etre que ducoup sous cette hypothese c'est pas le NTK qu'il faut voir
c'est pour ça que recentrer la discussion du NTK sur l'applicabilité, les hypothèses et le cadre high dimensionnal, high compute, high size est bien (sinon
ça devient trop difficile )

AFAIRE : à noter aussi les constantes de variance du ntk de terjek, c'est un résultat super dur

en fait la perspective du NTK quand m est grand et avec o3 o4 c'est de répondre pendant le training de sorte à pouvoir obtenir une optimization bound en fonction de L et M, car on peut étudie leur évolution directement puisqu'ils ne changent pas à un ordre supérieur près

il y a aussi un aspect important, ce n'est pas la depth qui change la structure profonde du spectre, elle l'élargi de manière linéaire
mais le spectre garde sa forme, on ne pourra jamais contrer le conditionnement ça c'est sur c'est écrit dans les équations
on pourrait peut etre calculer le conditionnement de la matrice de correction

d'ailleurs le 1er vecteur propre est PEUT ETRE celui des normes, à prendre avec des méga pincettes donc c'est celui qui dit qu'en cas d'homogénéité, c'est d'abord la norme qui compte (puisqu'elle est propagée) et donc d'abord on apprend quel scaling il faut pour chaque vecteur (en fonction de sa norme donc)
car c'est la "scaling appropriée de premier instance qui permettra ensutie de bien approximer la fonction f (impossible d'apprendre un relu à l'approximer
sans d'abord apprendre la norme), ça c'est ce que le NTK nous dit c'est qu'agumenter la profondeur ça marche, et qui scale bien en fonction de L
donc le pouvoir séparateur des normes (je suis pas sur de ca) augmente avec L, mais l'autre pouvoir aussi (faire un dessin du pouvoir separateur des normes sur une sphere, et pourquoi ça augmente, parce qu'on rajoute des chemins ?) ATTENTION JE SUIS PAS SUPER SUR DE CA
--- FAIRE DES EXPERIENCES NUMERIQUES SUR LA DEPENDANCE DU 1er vecteur propre en fonction des normes, pas evident du tout, mais secondaire pour une scaling law
mais on sait que le 1er vecteur propre depend du dataset avec une intégrale (en fnoction de N on peut avoir la correction pour operator discrétisé)


et après cela fait c'est la partie autre du spectre qui importe et c'est celle là qui nous intéresse, on veut apprendre les patterns
et là le ntk ne le décrit pas bien en fait, car le spectre ne s'approche vraiment pas du spectre qu'il nous faut, avec une separation
(à verifier avec MP, refaire calculs) des eigenvalue quasi identiques, d'ou l'idée d'une matrice random quelque part sur un sous espace

donc dans les nouveaux aspects mathémtiques à introduire pour analyser ça on a le narrow ntk, que l'on pourrait développer dans un aspect purement creatif aussi


ce qu'il faut faire :
traiter les correlations eigenvectors OK
verifier le MP pour les scalings inter valeurs propres OK

verifier les scalings dans o3 o4, puis verifier les scalings internes
implementer o4 en 
montrer une gueule de hessienne en comparaison avec le NTK

faire les liens terjek variance seleznova
release le code en mode n'importe qui peut telecharger et run, une fois les optimization faites

faire des diapos comme la soutenance, rapport comme la soutenance (car on est à la moitié, enfin commencer pour la semaine pro avancer, voir au feeling
si lex experiences disent uqelque chose et sinon dire que la semaine prochaine je faerfais aca)

pas oublier que les correlations qu'on calcule c pas grave à conjugaison pres (dans la base interne)

peut etre que la je vais avoir besoin de l'aide d'haizhao pour comprendre vraiment le truc de l'optimization bound
verifier pour d'autres densités non uniformes













en deuxieme instance : 
comment ils ont derive le surmrise ?

tester le NTK en Deep Super Relu Networks (see haizhao yang refs on google)
en fait otutes les experimentations on peut les lancer sur le tore et sur espace gaussien
il faudrait aussi traiter shiijun et rank mmfn ffnn, et aussi pour haizhao en exposant fractionnaire l'approximation
traiter aussi la dimensionnalité pour le sobolev training, la dependance du sectre en la dimension
traiter les experiences des autres codes pour voir ce qu'ils donnent, et traiter le code en jax aussi pour calculer o4 avec la hessienne du NTK
surtout le narrow ntk de ntk for deep narrow network (see ref on google)
ecrire la variable aleatoire du NTK en faisant la somme de chemin à la physicienne (ising ?)
euh l'aspect relu doit etre careful parce que un peu complexe la finite width comme dit par ethan dyer





todo :

- show all results compiled (derivations, implementations) for K3/K4
- show results for GOE statistics 



TODODO :
- Better implementation with jax
- contacer terjek car il a beaucoup d'idées en tête et est seul

questions : 
- 
- Time schedule ? office or shared desk ?
- Objective of publications of results
