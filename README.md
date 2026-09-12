# README:

## SuperBPE Tokenizer for African Languages

## 

### Dataset

Our dataset covers over 30 African languages compiled from open-access Hugging Face sources, including masakhanews, MasakhaNER 2.0 / MasakhaPOS, AfriSenti, AfriQA, Menyo-20k, SADiLaR, NLLB-Seed, MADAR / arabic-dialect-corpus, isiZulu news, Kanuri book corpus, Fulah sentiment corpus, Health translation dataset (English-Nuer-Dinka), Chichewa MT, bible zarma, Bambara raw text corpus, Akan sentiments corpus, and Wori-wolof instructions.

Languages span three morphological typologies:

* **Agglutinative**: Chichewa, Shona, Fula, isiXhosa, isiZulu, Kinyarwanda, Kirundi, Kiswahili, Lingala, Luganda, Oromo, Somali, Xitsonga.
* **Fusional**: Amharic, Tigrinya, Hausa, and Arabic dialects (Algerian, Moroccan, Egyptian, Tunisian, Libyan, Sudanese, Mauritanian).
* **Analytic**: Akan, Bambara, Igbo, Nigerian Pidgin, Yoruba, Afrikaans, Zarma, Dinka, Nuer, Kanuri.

To build balanced typology training sets, language\_concat.py was run twice per typology directory using seed 42 to generate two shuffled passes per category: agglutination\_1.txt, agglutination\_2.txt, fusional\_1.txt, fusional\_2.txt, analytic\_1.txt, and analytic\_2.txt. A unified benchmark test set (languages.txt) of 2,684,945 characters (2,639,704 atomic characters, 633,046 words) was created by sampling and globally shuffling sentences across all languages.

### 

### Training Pipeline

* **Preprocessing**: Raw text is sanitized using language\_concat.py. The pipeline strips URLs, social media handles, hashtags, PII, HTML tags, and promotional spam. Standalone numbers are removed to avoid vocabulary bloat, while Arabizi terms (e.g., 3amer, 7abibi) and alphanumeric technical terms (e.g., COVID-19) are preserved. Sentences are segmented using script-aware punctuation across Latin, Ethiopic, and Arabic scripts.
* **Model Architecture \& Inference**: 

  * **SuperBPE:** We implement a two-stage Byte Pair Encoding (SuperBPE) curriculum. Stage 1 applies intra-word BPE within whitespace boundaries to learn subwords, affixes, and stems. Stage 2 lifts whitespace constraints, merging adjacent tokens across spaces into multi-word superwords. 
  * **Tokenization:** tokenizer.py is programmed to find the shortest path based on a Directed Acyclic Graph (DAG) using backward dynamic programming (DP). With uniform edge costs (+1 per token), this guarantees a mathematically global minimal token count for any given vocabulary.
* **Hyperparameter Budgets**: The 20,000 token budget is partitioned across morphological families:

  * **Agglutinative** (35% budget / \~6,895 slots): 80% Stage 1 subwords, 20% Stage 2 superwords, max phrase length 2 words.
  * **Fusional** (35% budget / \~6,895 slots): 65% Stage 1 subwords, 35% Stage 2 superwords, max phrase length 3 words.
  * **Analytic** (30% budget / \~5,910 slots): 40% Stage 1 subwords, 60% Stage 2 superwords, max phrase length 4 words.
* **Key Design Choices**:

  * Affects different thresholds based on the language type (agglutination, analytic, fusional) so that each type is fairly represented.
  * Non-ASCII characters are converted to ASCII markers \[U+XXXX NAME] and masked to atomic placeholders \_\_U\_XXXX\_\_ during training so each marker maintains an atomic length of 1.
  * Rejects casing duplicates for superwords by keeping only the single most frequent casing variant.
  * Enforces leading-space formatting for superwords ( w1 w2) and discards unspaced duplicates.
* **Data Volume Observations**: Adding data up to two passes (\_1 and \_2) significantly improved vocabulary coverage. Adding further passes (3x, 4x) showed performance stagnation. Adding only one pass (\_1) reduced the tokenizer performance to 3.19 char/token on our test compared to 3.33 char/token for the other tests. Further scaling was halted due to time constraints and strong baseline performance.

### 

### Evaluation

Testing was conducted locally before evaluating on the competition platform.

1. #### Local Benchmark Testing

We tested locally using test\_superBPE\_v7.py on languages.txt (2,639,704 atomic characters, 633,046 words) to verify performance and track iterations:

* **Atomic Compression Ratio**: 3.33 characters/token.
* **Word Fertility**: 0.80 words/token.
* **Encoding Speed**: 837,523 characters/sec (200,852 words/sec).
* **Reconstruction Accuracy**: 100% exact lossless round-trip match (decode(encode(text)) == text).

#### 2\. Competition Platform Evaluation

On the platform's 2,000,000-character test corpus:

* **Runtime**: 0.43 seconds.
* **Tokens Emitted**: 8,944 tokens.
* **Calculated Compression Ratio**: 223.61 characters/token.
* **Reconstruction Accuracy**: 100% exact lossless round-trip match.

## 

### Reproduction

Execute scripts in this order:

**1. Generate Balanced Corpora per Typology**:

python language\_concat.py --input-dir raw\_datasets/agglutination\_languages --output-file agglutination\_1.txt --seed 42

python language\_concat.py --input-dir raw\_datasets/agglutination\_languages --output-file agglutination\_2.txt --seed 42

python language\_concat.py --input-dir raw\_datasets/fusional\_languages --output-file fusional\_1.txt --seed 42

python language\_concat.py --input-dir raw\_datasets/fusional\_languages --output-file fusional\_2.txt --seed 42

python language\_concat.py --input-dir raw\_datasets/analytic\_languages --output-file analytic\_1.txt --seed 42

python language\_concat.py --input-dir raw\_datasets/analytic\_languages --output-file analytic\_2.txt --seed 42

**2. Build Benchmark Test Corpus**:

python language\_concat.py --input-dir raw\_datasets --output-file languages.txt --seed 42

**3. Train SuperBPE Model**:

python superbpe\_v7.py --data-dir cleaned\_datasets --vocab-size 20000 --output tokenizer.json

**4. Benchmark and Verify**:

python test\_superBPE\_v7.py languages.txt --tokenizer tokenizer.json

## 

### Appendix

* **Team Members**: Wiam Atfi and George Aladejana (Team Bangweulu).
* **Mentor**: Elinah Moyo.
* **Data Sources \& Licenses**: MasakhaNER 2.0 / MasakhaPOS (CC-BY-4.0), AfriSenti (CC-BY-4.0), arabic-dialect-corpus (MIT), isiZulu news, Kanuri book corpus (CC-BY-4.0), Fulah sentiment corpus (MIT), Health translation dataset (MIT), Chichewa MT, bible zarma, Bambara raw text, Akan sentiments corpus (MIT), Wori-wolof instructions (CC-BY-4.0).

 

### References **:**



“

@article{Adelani2023MasakhaNEWS,

title={MasakhaNEWS: News Topic Classification for African languages},

author={David Ifeoluwa Adelani and Marek Masiak and Israel Abebe Azime and Jesujoba Oluwadara Alabi and Atnafu Lambebo Tonja and Christine Mwase and Odunayo Ogundepo and Bonaventure F. P. Dossou and Akintunde Oladipo and Doreen Nixdorf and Chris Chinenye Emezue and Sana Sabah al-azzawi and Blessing K. Sibanda and Davis David and Lolwethu Ndolela and Jonathan Mukiibi and Tunde Oluwaseyi Ajayi and Tatiana Moteu Ngoli and Brian Odhiambo and Abraham Toluwase Owodunni and Nnaemeka C. Obiefuna and Shamsuddeen Hassan Muhammad and Saheed Salahudeen Abdullahi and Mesay Gemeda Yigezu and Tajuddeen Gwadabe and Idris Abdulmumin and Mahlet Taye Bame and Oluwabusayo Olufunke Awoyomi and Iyanuoluwa Shode and Tolulope Anu Adelani and Habiba Abdulganiy Kailani and Abdul-Hakeem Omotayo and Adetola Adeeko and Afolabi Abeeb and Anuoluwapo Aremu and Olanrewaju Samuel and Clemencia Siro and Wangari Kimotho and Onyekachi Raphael Ogbu and Chinedu E. Mbonu and Chiamaka I. Chukwuneke and Samuel Fanijo and Jessica Ojo and Oyinkansola F. Awosan and Tadesse Kebede Guge and Sakayo Toadoum Sari and Pamela Nyatsine and Freedmore Sidume and Oreen Yousuf and Mardiyyah Oduwole and Ussen Kimanuka and Kanda Patrick Tshinu and Thina Diko and Siyanda Nxakama and Abdulmejid Tuni Johar and Sinodos Gebre and Muhidin Mohamed and Shafie Abdi Mohamed and Fuad Mire Hassan and Moges Ahmed Mehamed and Evrard Ngabire and and Pontus Stenetorp},

journal={ArXiv},

year={2023},

volume={}

}

“

 

"@inproceedings{Muhammad2023AfriSentiAT,

title={AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages},

author={Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Abinew Ali Ayele and Nedjma Ousidhoum and David Ifeoluwa Adelani and Seid Muhie Yimam and Ibrahim Sa'id Ahmad and Meriem Beloucif and Saif Mohammad and Sebastian Ruder and Oumaima Hourrane and Pavel Brazdil and Felermino D'ario M'ario Ant'onio Ali and Davis Davis and Salomey Osei and Bello Shehu Bello and Falalu Ibrahim and Tajuddeen Gwadabe and Samuel Rutunda and Tadesse Belay and Wendimu Baye Messelle and Hailu Beshada Balcha and Sisay Adugna Chala and Hagos Tesfahun Gebremichael and Bernard Opoku and Steven Arthur},

year={2023}

}

“

 

"@dataset{arabic\_dialect\_corpus,

title={Arabic Dialect Corpus},

author={Dataflare},

year={2026},

publisher={Hugging Face},

url={https://huggingface.co/datasets/dataflare/arabic-dialect-corpus}

}

"

 

"Alp Öktem, Muhannad Albayk Jaam, Eric DeLuca, Grace Tang

Gamayun –  Language Technology for Humanitarian Response

In: 2020 IEEE Global Humanitarian Technology Conference (GHTC)

2020 October 29 - November 1; Virtual.

"

 

"@dataset{fulah\_sentiments\_corpus,

title={Fulah Sentiment Corpus},

author={Mich-Seth Owusu},

year={2025},

url={https://huggingface.co/datasets/michsethowusu/fulah-sentiments-corpus}

}

"

 

"DigitalUmuganda/NMT\_Health\_parallel\_data\_en\_kin

"

 

"@dataset{akan\_sentiments\_corpus,

title={Akan Sentiment Corpus},

author={Mich-Seth Owusu},

year={2025},

url={https://huggingface.co/datasets/michsethowusu/akan-sentiments-corpus}

}

"

 

“

@dataset{diop2026wori,

author = {Diop, Marème},

title = {{WORI}: Wolof Reverse Instruction Dataset},

year = {2026},

publisher = {Hugging Face},

url = {https://huggingface.co/datasets/m-a-d-i/wori-wolof-instructions}

}

“

 

