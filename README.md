# pos-tagger

CS B551 Fall 2016, Assignment #3

Your names and user ids:
 Sarvothaman Madhavan    -   madhavas
 Raghavendra Nataraj     -   natarajr
 Prateek Srivastava      -   pratsriv

(Based on skeleton code by D. Crandall)


Training:
While training the following probabilities are calculated using the plug-in principle (plug in the counts/total
occurrence in place of probabilities:
1. Initial state probabilities            :   Out of all the sentences, how many times did each part of speech
                                             started the sentence
2. State transition probabilities         :   Count the occurrence of each pairs of pos that occurred in training data
3. Emission probabilities                 :   For each pos, count how many times any word occurred
                                             as that part of speech
4. Complex state transition probabilities :   Count the occurrence of each triple pairs of pos that occurred
                                             in training data.


Complex Model:
For the complex model, we started out with estimating the P(S1/W) and P(S2/W) as special cases since these do not have
the same structure as all other probabilities i.e P(S3/W). Once these two are saved as tau1 and tau2, all other
probability calculations will lookup previous tau values to estimate current "level" probabilities and further save
it as current level of tau

Posterior Calculation:
         For posterior calculation, we assume the HMM model

Accuracy Table for bc.test
--------------------------------------------------------
             |   Word Accuracy   |   Sentence Accuracy
--------------------------------------------------------
Simplified    |       93.96%      |       47.50%
--------------------------------------------------------
hmm           |       95.03%      |       54.05%
--------------------------------------------------------
Complex       |       92.61%      |       44.45%
--------------------------------------------------------

