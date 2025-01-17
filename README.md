# Amharic_Dependency_ParserApp
## Abstract
Dependency parsing provides information regarding word relationships and has many applications in natural language processing. Several methods of dependency parsing have been proposed in the literature for English and European languages. However, no sufficient dependency parsing system is available for Amharic. Due to its morphological structure and low-resource availability, customizing available dependency parser systems is not efficient for Amharic language.
In this paper, a dependency parser system is proposed for the Amharic language. In the proposed approach, an arc-eager transition-action classifier is trained on transition configurations generated from Amharic treebank to construct unlabeled dependency tree. Then, a relation-label classifier is trained on pairs of POS tags of the head and the dependent words to predict the label of the dependency relation.
Experiments were conducted on 1574 annotated sentences prepared and collected from UD-Amharic treebank. The classifiers were tested on 30% of the dataset, and 92% and 81% accuracies were found for the transition-action and relation-label classifier, respectively. The proposed system was also evaluated using an unlabeled and labeled attachment score, and 91.54% unlabeled and 86% labeled attachment scores were found. Our experimental results demonstrate that the proposed system can be used for parsing Amharic sentences and as a preprocessing tool during the development of natural language processing tools.


Full paper link:https://api.semanticscholar.org/CorpusID:248660135
}
