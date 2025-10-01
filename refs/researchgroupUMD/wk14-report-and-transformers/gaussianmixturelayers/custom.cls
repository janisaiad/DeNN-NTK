\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{custom}[Custom class]
\newcommand{\todo}{\color{red}}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[letterpaper, pmlr]{custom_jmlr}

\jmlrpages{}
\jmlrvolume{}
\jmlryear{}
\jmlrproceedings{PMLR}{}

\ifanonsubmission
 \newcommand{\coltauthor}[1]{}
 \author{author names withheld}
 \editor{Under Review}
 \jmlrworkshop{}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\coltauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{}
  \jmlrworkshop{}
 \fi
\fi




