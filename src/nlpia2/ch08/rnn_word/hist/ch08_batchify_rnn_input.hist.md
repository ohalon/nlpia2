
```python
>>> %run generate
>>> hist
>>> who
>>> %run main.py
>>> train_data
>>> who
>>> corpus
<data.Corpus at 0x7f3082d87040>
>>> corpus.train
tensor([ 4,  0,  1,  ..., 15,  4,  4])
>>> cd ~/code/team/exercises/
>>> %run generate
>>> who
>>> generate_words(model=model, vocab=vocab, prompt='He')
>>> import torch
... from data import Corpus
... from generate import generate_words
... from model import RNNModel
... 
... corpus = Corpus('data/wikitext-2')
... vocab = corpus.dictionary
... with open('model.pt', 'rb') as f:
...     model = torch.load(f, map_location='cpu')
... state_dict = model.state_dict()
... 
... model = RNNModel('GRU', vocab=corpus.dictionary, num_layers=1)
... model.load_state_dict(state_dict)
... # ' '.join(generate_words(model=model, vocab=vocab, prompt='The', temperature=1))
...
<All keys matched successfully>
>>> generate_words(model=model, vocab=vocab, prompt='He')
['.',
 '=',
 'Robert',
 'Mosley',
 '(',
 'Centipede',
 ')',
 'has',
 'led',
 'a',
 'living',
 'stone',
 'of',
 'his',
 'mirror',
 'and',
 'by',
 'the',
 'vegetables',
 'were',
 'given',
 'his',
 'work',
 'to',
 'work',
 'during',
 'the',
 'Republic',
 '.',
 'He',
 'uncovered',
 'sought',
 'to',
 'have',
 'a',
 'police',
 'first',
 'Cessna',
 'before',
 'he',
 'chronicler',
 'one',
 'January',
 'his',
 'similar',
 'representative',
 ',',
 'in',
 'the',
 '3rd',
 'Municipal',
 'inventor',
 'and',
 'wielded',
 'state',
 'too',
 'far',
 ';',
 'in',
 'this',
 'group',
 'belonging',
 'to',
 'the',
 'country',
 ',',
 'clergyman',
 'was',
 'escaped',
 'to',
 'Colonels',
 'to',
 'compete',
 'into',
 'attachment',
 '.',
 'Edward',
 'was',
 'the',
 'school',
 'church',
 'made',
 'building',
 'of',
 'the',
 'European',
 'Persian',
 'Political',
 'asci',
 'championships',
 'because',
 'prohibits',
 'motor',
 'Marshal',
 'was',
 'and',
 'did',
 'not',
 'required',
 'prize',
 'killer',
 '.',
 '=',
 'He',
 'were',
 'mainly',
 'failure',
 'to',
 'twenty',
 '@-@',
 'further',
 'rectangular',
 'Gefion',
 'on',
 'nine',
 'hundred',
 'engines',
 'and',
 'has',
 'officials',
 'fulfill',
 'bent',
 'shooting',
 'from',
 'their',
 'three',
 '%',
 'of',
 'terms',
 '.',
 'operations',
 'that',
 'were',
 'took',
 'third',
 ',',
 'and',
 'they',
 'helps',
 'HNC',
 'as',
 'Polish',
 'effort',
 'to',
 'build',
 'for',
 'deadly',
 'runners',
 '@-@',
 'up',
 '.',
 'The',
 'raised',
 'portal',
 'of',
 'Taylor',
 'was',
 'appointed',
 'by',
 'worked',
 'on',
 'his',
 'skills',
 'being',
 'discussions',
 '"',
 'his',
 'own',
 'chance',
 'at',
 'a',
 'range',
 'with',
 'a',
 'live',
 'value',
 '.',
 '"',
 '=',
 'Walpole',
 'took',
 'the',
 'title',
 'station',
 'to',
 'reach',
 'his',
 'administrative',
 'car',
 'as',
 'without',
 'Lawrence',
 '.',
 'A',
 'serious',
 'findings',
 'reminiscent',
 'of',
 'Hum',
 'and',
 'Kentucky',
 '(',
 '1160',
 ')',
 'of',
 'this',
 'start',
 'rarely',
 'varied',
 'for',
 'in',
 'an',
 'extreme',
 'Scottish',
 'Ambon',
 '.',
 'His',
 'constituency',
 'of',
 'sending',
 'a',
 'Advent',
 'Rufus',
 'was',
 'Nicholson',
 'despite',
 'that',
 'claims',
 'to',
 'struggle',
 'anything',
 'with',
 'them',
 'made',
 'increasingly',
 'former',
 'Africaine',
 ',',
 'with',
 'Nash',
 'other',
 'and',
 'religious',
 'projects',
 '.',
 'The',
 'proposed',
 'sold',
 'allegiance',
 'to',
 'complete',
 'had',
 'allowed',
 'to',
 'Ipswich',
 'Newark',
 'in',
 '20',
 'Aires',
 '.',
 'It',
 'is',
 'unknown',
 'because',
 'the',
 'Annals',
 'of',
 'Jonathan',
 'people',
 'at',
 'Centrosaurus',
 'in',
 'favour',
 'of',
 'an',
 'Automobile',
 'centre',
 'of',
 'her',
 'salary',
 'and',
 'buses',
 '.',
 'Causes',
 'of',
 'the',
 'starboard',
 'three',
 'hours',
 'is',
 'rescued',
 'by',
 'many',
 'of',
 'the',
 'year',
 '.',
 'In',
 'the',
 'region',
 ',',
 'Mosley',
 'was',
 'an',
 'epic',
 'in',
 'turn',
 'from',
 'which',
 'Given',
 'Coleman',
 "'s",
 'UN',
 'and',
 'Owego',
 'of',
 'his',
 'soldiers',
 'consider',
 'but',
 'the',
 ',',
 'explaining',
 'their',
 'man',
 'into',
 'political',
 'or',
 'when',
 'the',
 'audience',
 '.',
 'The',
 'oldest',
 'producer',
 'were',
 'imposed',
 'in',
 '1954',
 '–',
 '61',
 'platforms',
 'to',
 'Milton',
 'leader',
 'Devil',
 'and',
 'Polo',
 'against',
 'removing',
 'outstanding',
 'draft',
 '.',
 'He',
 'received',
 'the',
 'meantime',
 'in',
 'his',
 'wife',
 ',',
 'a',
 'member',
 'of',
 'Ticonderoga',
 'in',
 '1910',
 '.',
 '=',
 '=',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 'Taylor',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 '=',
 '=',
 'Causes',
 'sponsored',
 'Odaenathus',
 'and',
 'even',
 'a',
 'elephant',
 'after',
 'he',
 'wants',
 'to',
 'prevent',
 'a',
 'space',
 'skill',
 'during',
 'the',
 'house',
 'of',
 'Li',
 '.',
 'General',
 'screenwriters',
 'of',
 '1864',
 'as',
 'only',
 'to',
 'kill',
 'the',
 'lesser',
 'Roman',
 'description',
 'he',
 'collaborated',
 'with',
 'nominate',
 'the',
 'sons',
 '.',
 'However',
 ',',
 'having',
 'a',
 'soon',
 'pitch',
 ',',
 'and',
 '"',
 'king',
 'Bennett',
 "'s",
 'father',
 '"',
 'is',
 'involved',
 'by',
 'Blanche',
 '(',
 'legal',
 ')',
 'and',
 'that',
 'occasionally',
 'entertainment',
 'artistic',
 'infections',
 '.',
 'The',
 'kerb',
 ',',
 'the',
 'court',
 ',',
 'and',
 'Monroe',
 'and',
 'in',
 'the',
 '1980s',
 'was',
 'underway',
 '.',
 'a',
 'rotating',
 'design',
 '.',
 'His',
 'life',
 'is',
 'dark',
 'of',
 'as',
 'a',
 'moral',
 'cap',
 ',',
 'whose',
 'events',
 'may',
 'have',
 'been',
 'found',
 'when',
 'there',
 'are',
 'five',
 'starling',
 'replaced',
 'by',
 'illness',
 '.',
 'He',
 'formalized',
 'lawlessness',
 'modeled',
 'with',
 'a',
 'target',
 'style',
 'does',
 'the',
 'endemic',
 'to',
 'form',
 'for',
 'combined',
 '.',
 'Numerous',
 'release',
 'was',
 'renew',
 'to',
 'be',
 'tens',
 'of',
 'longer',
 'amounted',
 'to',
 'treat',
 '39',
 '%',
 'show',
 '.',
 'The',
 'Japanese',
 'takes',
 'distinct',
 'stronger',
 'and',
 'considerable',
 'Ptolemaic',
 'of',
 'Ireland',
 ',',
 'and',
 'his',
 'subject',
 'allowing',
 'the',
 'songs',
 'for',
 'both',
 'beyond',
 'a',
 'sentimental',
 'years',
 ',',
 'under',
 'the',
 'Liberals',
 'of',
 'domain',
 'as',
 'being',
 'longevity',
 'jams',
 ',',
 'and',
 'PSH',
 'Lewis',
 'thus',
 'must',
 'have',
 'been',
 'till',
 'Latin',
 'Ireland',
 ',',
 'each',
 'of',
 'the',
 'insular',
 'and',
 'was',
 'thought',
 'of',
 'pool',
 '.',
 'However',
 ',',
 'he',
 'was',
 'chosen',
 '1900s',
 '.',
 'The',
 'engineers',
 'was',
 'published',
 'in',
 'the',
 'ninth',
 ',',
 'often',
 'suggested',
 'for',
 'dead',
 'funds',
 ',',
 'but',
 'falling',
 'sexual',
 'police',
 'frequencies',
 'on',
 'devices',
 '.',
 'On',
 'July',
 '10',
 ',',
 '2008',
 ',',
 'the',
 'silver',
 'and',
 'king',
 'Independent',
 'Council',
 'is',
 'a',
 'small',
 'resistance',
 'from',
 'the',
 'war',
 'cell',
 'It',
 'is',
 'required',
 'all',
 'of',
 'Peronism',
 'himself',
 'and',
 'Connor',
 'to',
 'commemorate',
 'Britain',
 'and',
 'humane',
 'NCAP',
 '.',
 'According',
 'to',
 'ribosomes',
 'in',
 'the',
 'Persian',
 '@-@',
 'genetic',
 'junk',
 'rose',
 'that',
 '"',
 'field',
 'Stronger',
 'fill',
 '"',
 'an',
 'operation',
 'competition',
 '.',
 'Odaenathus',
 'also',
 'recovered',
 'on',
 'Metro',
 'gunfire',
 ',',
 'now',
 'cleared',
 'number',
 'of',
 'dates',
 'of',
 'calligraphy',
 'in',
 'early',
 '1991',
 ',',
 'and',
 'four',
 'understanding',
 '.',
 'The',
 'participant',
 'also',
 'showed',
 'three',
 'report',
 ',',
 'and',
 'as',
 'some',
 'of',
 'practices',
 'that',
 'they',
 'could',
 'be',
 'special',
 'leaders',
 'of',
 'terrifying',
 'bodies',
 '.',
 'Many',
 'seawall',
 'for',
 'Faulkner',
 "'s",
 'laboratory',
 'were',
 'extremely',
 'sufficient',
 'paid',
 'and',
 'that',
 'bound',
 'heavily',
 '.',
 'About',
 '25',
 'million',
 'members',
 'may',
 'be',
 'buried',
 'between',
 'birch',
 'and',
 'government',
 ',',
 'an',
 'interest',
 'of',
 'Abingdon',
 'and',
 'areas',
 ',',
 'Croatian',
 'deity',
 '(',
 'a',
 'study',
 'on',
 'them',
 'and',
 'pass',
 'to',
 'Dukes',
 'of',
 'Dublin',
 'Shula',
 ')',
 '.',
 '=',
 '=',
 'Valkyria',
 'Valkyria',
 'Etymology',
 'of',
 'the',
 'Primrose',
 'Valkyria',
 'Valkyria',
 '=',
 '=',
 'Note',
 ':',
 '=',
 'The',
 'Wii',
 '(',
 'Broncos',
 'abundance',
 'of',
 'violacea',
 ',',
 'probably',
 'later',
 'related',
 'to',
 'sexual',
 'bodies',
 'school',
 '.',
 'Accordingly',
 ')',
 ',',
 '30',
 '%',
 'of',
 'Ireland',
 'may',
 'be',
 'used',
 'for',
 'an',
 'than',
 'exposure',
 '.',
 'Time',
 ',',
 'the',
 'MCC',
 'has',
 'also',
 'been',
 'studied',
 'into',
 'a',
 ',',
 'meaning',
 'in',
 'its',
 'early',
 'Neolithic',
 'birds',
 'during',
 'extra',
 'trials',
 '.',
 'At',
 'the',
 'time',
 'of',
 'contemporary',
 'endangered',
 'word',
 'a',
 'war',
 'modified',
 'there',
 'is',
 'ingenious',
 'that',
 'it',
 'is',
 'particularly',
 'an',
 'slave',
 'plateau',
 ',',
 'of',
 'wines',
 ',',
 'and',
 'they',
 'are',
 'fixed',
 'subordinate',
 '.',
 'Exhale',
 'in',
 'Russia',
 ',',
 'ranking',
 'curved',
 'than',
 'the',
 'province',
 ',',
 'lookouts',
 ',',
 'stoats',
 ',',
 'fetus',
 'especially',
 'al',
 'competitions',
 '(',
 'under',
 'of',
 'Shell',
 'p',
 ')',
 'than',
 'a',
 'product',
 'off',
 'and',
 'the',
 'Luftwaffe',
 'Caishi',
 '.',
 'It',
 'has',
 'placed',
 'up',
 'and',
 '18',
 'mm',
 '(',
 '7',
 '@.@',
 '2',
 'by',
 'Cougar',
 'Gurion',
 ')',
 ',',
 'an',
 'grass',
 'of',
 'mid',
 '@-@',
 'footed',
 ',',
 'on',
 '1',
 'March',
 'and',
 'one',
 '.',
 'The',
 'terreplein',
 'tends',
 'to',
 'have',
 'been',
 'powerful',
 'comprising',
 'the',
 'next',
 'the',
 'other',
 'risk',
 'in',
 'an',
 'eye',
 'mi',
 '(',
 'the',
 'base',
 'of',
 'Saginaw',
 ')',
 ',',
 'the',
 'city',
 '.',
 '=',
 '=',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 'in',
 'basement',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 '=',
 '=',
 '=',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 'Running',
 'apart',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 '=',
 '=',
 '=',
 'Valkyria',
 'Valkyria',
 'Themes',
 'and',
 'Bailey',
 'Valkyria',
 'Valkyria',
 '=',
 '=',
 'Yeovil',
 'Valkyria',
 'Valkyria',
 'Valkyria',
 '=',
 'It',
 'that',
 'not',
 'grow',
 'around',
 'its',
 'ancient',
 '1901',
 '.',
 'The',
 'inscription',
 'is',
 'unwilling',
 'to',
 'Asylum',
 'making',
 ...]
>>> generate_text(model=model, vocab=vocab, prompt='He')
'is outside with the fisheries of a ( Harold ) ( to this point of many @-@ studded towards the 1845 ; a third doctoral registered rate that later ) . = = Valkyria Valkyria Valkyria referendum – 1553 Valkyria Valkyria Valkyria = = Hornung agreed to offer having towed not sitting away . = He avoided his mother could comprise against journalist , whose assistant was relegated to manage to keep left sides from to salsa programme the game of the 3DS in the estuarine . completed the film youth ; A heaviest depressed named Cardinal @-@ kun ( Francisco Earl ) . The first sister eldest , which they follow the visions of the name = Nevertheless of 1935 , a Roman school Fei \'s former Flanders remains in Ireland driver Department of the Pre @-@ colonial building ; he was later excessive . Its last the performer wrote in the brawl and an rendering that they were only free to deal and Jean school , Hunter was . Swing 1782 toured from 1960 . He also returned to the four for many his divisional family , was originally used to be a reasonable @-@ @-@ century case in the 1920s . He had lived the WHO event on the ground @-@ Borders hope of a Futa U @-@ defined afterwards . Lincoln business abandoned the issues with the stress and magician president and Tecolutla Jurchen Song methods well @-@ based into on a meeting trolley for a one @-@ nine installation . Lützow was suggested , with the Accordingly website Johnson himself , stated , that delays was of Appeals . In some ways , practice began due to the nutritional Norman- , cease in order to use if the impoverished universities . Walpole spent almost " map as Marlon " . Coliseum was in the second residence in total stages of his reputation to have was approved by youth and left . Seattle tells the group of art , where it was characterized by " F. " , Edward Archer \'s Dog ratings from or Danielle reviews . At their considerable link bread guilty , Richard Litter Jones argues it , he owned into a teacher \'s operas of Crush and Bull . Former Sonthi Stone ( the poem of his Starring subtypes Mirror , is ) in the chief of Mexico , rather than quality of people . In the way of archaeology as Lady underground tourist Palmyrene brother F. an outstanding power \'s doctor @-@ based year . He \'s defense were consider by in 400 years of the Arts . He after Chen maintain describe whether the Ajuran pedestal of \'s first future descamisados heard when his Concerts king describes his seat against Philip . He acknowledged they served as a private @-@ classic of progressive and demolished . = Hornung wrote a " Kamara of opponents watching " senior and 1990s , referring for his enroll , the player , they were handled by professionally done with his privacy . = Supporters pose Craig his work on a memorial of his wife pleaded and the hymenium is a observations of \' ( ) with the Morrison \'s diary , as a cold expression of the Latin subcontinent . He has been widely paid as scholarship and religious acts who brought Dry prepare . In the company , Toirdelbach led by Alfa \'s leader files , probably once elevated toward Enterprise and punishment . " They that Vaudeville , ground and in the violence of life will to have seen " this work and would number his two shown . = The official title is the casting @-@ nature of the patron of and acute Etty \'s him as a girl . " As a specimen of " Office of the second circles and " Dress " for producing Europe and McMaster \'s tradition , Nichols gave the " Videos Deputy " . The Nazis became over both commencing ( Maneater ) with State particles to the beard . This thing that has been attempting to collect six main and nature . = The names incorporates a kitsune and blowing damage , specifically to become red as well . Llosa Gabriel as preserved commended including the power of David la dark to prey . = In the stage , Zimmermann the kakapo was able into Dilke valens . It was , however , the book was confused in a sublimation of the Gothic , including Ceratops princess , was off by The coffee institution as tradition \'s species as ensue during year . Its public controls also managed to Ames people of Cuban indie old commandment . = = Valkyria Valkyria Valkyria Protein and Ito Valkyria Valkyria Valkyria = = Foundation , who wrote that they are more than too late % , but mothers ( according to and postremission that subplot having been expected to both agility and feedback ; a vowel championships is fed with a extensions of cheilocystidia to Cai cream . The female @-@ inspired , each are by a young manner and their male varies from the county trees . The field series be attributed to contrast for the player and , in combat . One miner will be measured at a junction by a wood . Allah can be an amazing engine , but are rarely damaged on 3 Ma ( low – 21 to 6 ° C ) on its original body , absolute chicks , and the male plagiarized as a Nightingale , an orbital distance Jesus and also lauded each brutality . The kakapo continues to be a 171 and one kilometre ( but possibly now with heat nayavāda in a genus ) . = = = Valkyria T. Lacy Review Valkyria Valkyria = = In 1851 ( 2 – 7 ) , the Art Sweet Site ( modern status for the final exit edge ) is located from the first in damage , which they might be reducing by separate eight speckled earlier and was that emissions in the 6 @,@ 000 @,@ 000 candidates caused missiles'
>>> generate_text(model=model, vocab=vocab, prompt='The')
'KPA on October 27 , 2010 . It was collected into an irregular role in claims to speak of the Free , Mark and four of 2016 . On no arm other @-@ ground games , abandon the poll of inferior broad system , military canons since Lord O \'Malley gave the most trading classification . WASP @-@ spaced Mk planes also won rammed opposing Young in the spring , for a minimum lambs . = Mike Copland may be stopped for the table at the NBA Republican graced for 1000 short or notes . In order players of expectations , some three years threatens at Banksia BBC II called a crowd Boston . = goes to the UK in 1995 , Manor reached his beneficial at 21 % of the year . = = Valkyria Valkyria Geography Valkyria Valkyria = = England , there meant that both of the additional 2011 thereafter should lead a wider effect without Grand \'s successor . Although a particular @-@ sample judges occurred in 1987 , which includes the Blue County stemmed for Port @-@ au and counties ; there was 28 % , leading 4 % of 44 , 10 @.@ 10 % . After the second two years , the sunlight also received on 15 April 1933 , which noblemen in where Intent adult writing party arm ; researching although Augustan stations in the Interaction in a lurid style crossed between times to the international school \' capability . = = Valkyria Valkyria Lon location Valkyria Valkyria = = On November 5 , Lock Report on May 16 . His last shows were known by alleging Krypton crime Dublin . = On 8 August , fifteen passengers aired in US Torah , including 789 and international rejecting opening game . It also also viewed a prison for Gacko International and Ulster of alphabets . Advertising populations from African concentrates , use centres permitted to swing , using land , while league was not not immediately stalling . = = Valkyria Valkyria History Valkyria Valkyria = = = Valkyria Valkyria Pre textiles Valkyria Valkyria = = Odaenathus Parliament described the royal name ) and not tortured Carlisle @-@ Kensington , but his discovery is clearly for seasonal to a powerful woodland . This was a fundamental actress . But Watson discussion initially only . Description tried that a challenge for role , has the words and " genres of Copperfield . " whereas this is proper , bands that The nomads dismantled that are Brown , who has been supposed to be determined to have been released between the city . Odaenathus \' defense and with a speech Smile Baker mac believed his first singing had directed by two Moriarty succeeded ( titled One of letter : Doctors / 8 : rural H.G. Howard d ) , it is part of a synonym , who sees the whole genus regional editions . = After electrolysis broke by a agent despite overlapping blood spelling the arrest , The Irish Republic , outbreaks operated by the temple " acoustics " than no the common forces dating from teens on the surrounding World Operation 1650 . Some works of Ithaca from the term are towards teachers , which allows the themes of eminent workers and Norman : This union could not sink , despite being studied . In drugs , Chung increases titled Fornebu and Most with which is absorbed within melting , is , non @-@ General published , at Ceres contrasting it in the prose . The family invites continued to sometime to the exploding and eye entirely . = = Valkyria Valkyria Valkyria 448 Valkyria Valkyria Valkyria = = The kakapo Extinction attested ( in the 1950s respects ) , of the 1920s and Brittany fields in mature century . The noisy Thoth is predominantly recaptured in the 16th century , which is the most occupies truth , but more than knowledge of 40 ° ( shaped may have been Erastian announce ) bound to Muslim of luminosity , near nearly his largest administrative Ideal . As of space , it has slightly classified by virtually this wing itself to add a dark intervals a lifelong abundance species . If it has been seen as large terminated of Skye . = The island , Muganga and the fine valence substances of the orbiting proteins curves to form of wetlands . Political 28s must be Canadair , but construction is one nest spectrum as making serious view . This mass bones are reported between the age that day . It is seen in better towards a total of music of halides . After barbettes to the kakapo may be afraid during this cell and them , they can be measured to them as an spirits , boots for relatively useful into to London . Other plants are in statues , as an , with the mixing stars for his patch to promote colouration , but taken with certain a typical physical molecules . The presence of Pinkner is characterised about the kakapo is in its queen in possible Galim , churches or Pld ( hind neck ) . = Common starlings has been found near suggests to this schools usually applied to that their construction has personas discernible at least long . The fungus was one data of the common starling during the Deep mentality , especially or safe industrial than the west of the area of five . = = Valkyria Valkyria Valkyria Buddhists and 1054 Valkyria Valkyria Valkyria = = In 1891 and some starlings afford to a Christian aesthetics , Venus were related to the Dubliners . The case of penis is a Protestant portion of the cave . = = Valkyria Valkyria Architecture Valkyria Valkyria = = = Valkyria Valkyria Valkyria Valkyria synthesis in Public History Valkyria Valkyria Valkyria Valkyria = = Schools Marine kakapo are larger , while according to the clinical Johns ballet include T. Wood , Latin United River , and may have been in Andhra Coast . In Syria , chicks have been too less as well as small'
>>> generate_text(model=model, vocab=vocab, prompt='President')
"and images of the United States Navy Park . The Derfflinger and collapsed of his Japanese occupation would not be murder of his military universe , and the second time ' Old Union the Art provinces of America was for $ 26 million , death , Ico conscientious camps . = = Valkyria Valkyria History Valkyria Valkyria = = The Buddhist international political starling would have further pounds of Ireland , the Irish line — a PDT of Ireland ; to the highest Hidalgo by five instruction , the Barry River , executing that year , comedian Francis Byway that the building are been distinguished separately for two Wives feed , Others despite exposed to shoot , to legislation attempted to cross @-@ rich in the country . In his place in the platform , the kakapo 's rest of the unrest Jitsu ( Welsh De Points ) were proposed recorded by a more amateur church . Ireland is high body status . = Pine name Council in the wild designs his Many of mixing and drinking Ireland ; Ray Eno was heavily joined as his old dialogues during the year . The Bertin was divided into legislative major movement on six hours . The species of Wales , and began a same project , which competed as a touchdown . Recurring produced taught of the Uttar and then travels in Ireland , leading city through Frankfurt ; as the military publication that chemotherapy at FITs , motor Shah considers Ireland for the ongoing Conference later ( the center ) of initial cleaning and for large , and increased to farming glowing in the 16th century . = In The Atlantic 18th century to the port of all cases generally naturally scholars . In a way of the the Dania , 1850s and reign of Q. bowl , Katherine Water female expressed produce direct nuclear from sites 90 : gas early the significance among his rape faster , who can be classified as displays in NY 44th and Israel . There were strong work for their 100 fall , including the most like high and fault or shapes gate of few years , while in a GA of Trypanosoma is forced to only in unprotected , though an earlier apprentice Diggle would also have been established in operations . In late June 1898 Nos. December 27 . spectra of group 's 3D language cross the robins , and the privacy were threatened to wield at curiosity distinctive on role . = = Valkyria Valkyria Background Valkyria Valkyria = = There is no strategic , or defined shaped offerings cross against . This day is probably to be a Two @-@ less nucleus , although , does not be trapped , and could be purchased for the distantly appearance . It is nine species to Plunketts Creek , possibly introduced from 4 : an assembly ( Shannon ) on the interior . From police specimens are one point on August every to mention countries , though the latitudes State may Serbia 346 considerably that proper history with other in suburban confined to sight and hence multiple incursion eggs and of mammals . Because of some young financed Expressway nests from them is on some emphasizes , the lowest @-@ challenge each group is also being greater judged . The strength substantial similar to their sides are reopened . This stalks is forced to chemical helicopters , even and common starlings on other factors in line . This also authored this period of contracting temperature and methanide or cousins from this beds , which they have uncertain Geller . = Over the capital of the Plantagenet is possible in daylight , about less than only the floor fighting lists . Polls is shared with its influence that extremely catwalk occurred . This cluster can be in the Shrine of the face universities in those and dungeon . The ears are also capable of teenagers in grooves , where he comprise it enacted much in the tips as the new glow carved and defending birds . The pattern undermined being the next range of players and boss the ill pistols and interspersed their castle into the Modern low and north @-@ side . It has already received positive reviews , while engender most restrictions in part of the region and particularly for the Talking @-@ brown dads is both triangular and gray Native opposing , in the Shropshires . However , and grassland ( e.g. ) , ( ) , the oil in high species ; a chapterhouse may need in display as £ 3 @.@ Cardinal view , the favorite are able to sits on large confirmation , either fins and trees . Other different ability are anger northeast damages in Alaska groups ( C ! ) , flattened ( Minervois ) and personification are dripping in and two weight as custom females . It has been debated , although and it is broken from electron birds as provided as black tonnes of bird , rarely for Tessa . declaring the Asus the next weight , male will go to a nomen layer bid . stalk , with example , despite a maturity glad invasions in winter perspective , very transits of Zimbabwe affirmation . The female is less serious or flying under a transit of Childs proclaimed trouble . It is Lalich of driving on 3 @-@ order , registered with 5 % are the closely 978 @-@ shaped on # 7 ( hard to regions ) . A burn magnitude of mg species are seen on the walls day . It occurs at more verify shaft with his ideal surface . The derivative act is currently close to have a predator source , a female and occurs , leaving which is the collapse of a kakapo is a long point . He killed Marcell 308 to one female funding . In particular , its old period may appear as rich to take up strips of continental a sounds . It is currently in other conditions , because when Richard Johnson ( 127 ) ,"
>>> generate_text(model=model, vocab=vocab, prompt='The president of the United States')
', which was a named African crisis Treaty of Mexico as implying it . Seven two @-@ year was later enjoyed he signed making a down Wally McCarthy already husband . Rhode Rederi was stationed in the Los Angeles , early 20th century , , after the trademarked to end the other had the nation same proved to pay him . In of the Jin Rovers , Robert cirrhifer consisted of this time , including the form the past subalpine in the . = In the winter , with a funniest , Wembley , 1880 interests of a total @-@ game mound / CPS lead by a served with the language agency prevalent in New York . Mosley has already been appointed Le First immediate standards of draped . Despite the labor show Mary showed a crime controller the propeller Holmes from geographer IBM by a Georgia spot and each five of these members . After fact that 35 % from water rejecting coalition tour , Pitman allowed a strong role of public opinions . The mid @-@ school kings , relationships the of the fleet for the field , 33 homosexuality , remarkable whom it was now , but was carried @-@ than £ $ 1 @,@ 000 . The own book was went on a 10 @-@ year , a year based in features . Based after the center of contemporary weapons , including college communities , , Wiśniowiecki of Taiwanese Criticism levels of Ireland , who has been taken from the addition to develop the team . Both Mars , a national Jewish community opened in 2010 , the Reagan issued relations between Early , and wounded , the kakapo was now somewhat the stalk , and New Zealand and assassins were based at the Swiss Forest number of the parish in fifteen years . = = Valkyria Valkyria Valkyria Yorkshire Valkyria Valkyria Valkyria = = Early ambushes in mid @-@ Prince in captivity are in 1972 . In rainforests , meanwhile , Glass Pagoda on May 10 , diverse both Jun in France \'s , led overseas recovery for ritual two leadership of Australia , describing them habitats in the sexy \'s mark to . However , there were called Arabic , and loud transformation . As cultural Concession , the refusal . After a lumber subordinate when the Elder and der Julian Red Middle quadrupedal , calling that the eukaryotic Protection campus were the region of the 19th century , showing a trip from the college century . = Jesus awarded an Moldavia that he was split by direct major when died . The bell of downturn can be brought , . = A Dai lets vintage Technology , was still rapidly brings during the mid @-@ Russian coup ( – blue Husband ) . The Republic are expected to have scattered bishops . = = Valkyria Valkyria Banksia Valkyria Valkyria = = All @-@ score arrived created on women and few @-@ two genes with the Great Somerset . = The neighbour Calvert cyclones cats for the last of the Boeing reconcile Round that their purged majority of them in private Castle . The body was distinguished around Mogadishu , developed by Hairan stations for valence . Making during October 1942 ; the 13th @-@ century , Nebraska , always the city entered the United States , received by Vegas and Minnesota , signifying this involved title Site . prefers Peh ceratopsians and non @-@ August many tasks and in Europe and hindering were formed by long . Aerodrome Roman Hornung reported dedicated to gain possible library @-@ old motion to encounter the unit by the Industry School . The colony \'s population purchased Daspletosaurus on = requirements that dentists , Dallas Du Colorado , they have good many times by a three @-@ bank PS2 model during order to take music years . Jainism , the exact name was interned from the Colorado Mexican and the Government conducted . Under three hours in the Canary transit , two another programme were joined by Entertainer and animals . The potential Grade set unfounded nominated on 14 was to Riekki . Rulers a circle of walking with ancient engineering activities of Ireland while remaining revolt is firing of the Manas protests in the city on Le 1630 . In the upper protecting redeeming , withdrawing a wide neural winds of an Liberal armament on Earth , and otherwise enlisted recent Balbás by the 18th century , but is part of the king \'s food . It was published by London , a psychological source of masonry and as Hornung tearful in the Ganges between the island . Of these novels , his health Guy curb exercised , meaning that dies stonework is the most successful stain in the Western English events of Catholic Reformation . The island reopened of historical in settlers , residing and Ryūjinmura , Henry des des ( Russian % of , Teachers \'s ) , the , Emperor Hamilton , and the American Egyptian Rivers , which consists " is a live celebrations or living Nasor and similar in 1962 and the history of Ireland \'s reign ( in the subfamily del sacred Management ) . = = Valkyria Valkyria History Valkyria Valkyria = = It is sometimes consisting of the novel recorded in Away Raton , Ireland , where was written by 3 @.@ 5 × 37 ft ( 22 mi ) of disasters . Grey has performed a four @-@ war ocean species date on noon , may be still impossible , to be both Morhange in one reason . This bird is made as its rushes and shape " Allāh " , who will have been green for various involvement in his crown . The Witnesses made the variety of extreme strength on a most agreement of the tiring of the emperor . = In the late 1970s to his real ground , its King I are leads to grant of the mouth with hard and Ímar . The @-@ frequency pattern is , and even sentimental sonne to the group of'
>>> generate_text(model=model, vocab=vocab, temperature=.2, prompt='President')
>>> corpus
<data.Corpus at 0x7f306fb75730>
>>> corpus.train
tensor([ 4,  0,  1,  ..., 15,  4,  4])
>>> [vocab.idx2word[i] for i in corpus.train]
>>> [vocab.idx2word[i] for i in corpus.train[:100]]
['<eos>',
 '=',
 'Valkyria',
 'Chronicles',
 'III',
 '=',
 '<eos>',
 '<eos>',
 'Senjō',
 'no',
 'Valkyria',
 '3',
 ':',
 '<unk>',
 'Chronicles',
 '(',
 'Japanese',
 ':',
 '戦場のヴァルキュリア3',
 ',',
 'lit',
 '.',
 'Valkyria',
 'of',
 'the',
 'Battlefield',
 '3',
 ')',
 ',',
 'commonly',
 'referred',
 'to',
 'as',
 'Valkyria',
 'Chronicles',
 'III',
 'outside',
 'Japan',
 ',',
 'is',
 'a',
 'tactical',
 'role',
 '@-@',
 'playing',
 'video',
 'game',
 'developed',
 'by',
 'Sega',
 'and',
 'Media.Vision',
 'for',
 'the',
 'PlayStation',
 'Portable',
 '.',
 'Released',
 'in',
 'January',
 '2011',
 'in',
 'Japan',
 ',',
 'it',
 'is',
 'the',
 'third',
 'game',
 'in',
 'the',
 'Valkyria',
 'series',
 '.',
 '<unk>',
 'the',
 'same',
 'fusion',
 'of',
 'tactical',
 'and',
 'real',
 '@-@',
 'time',
 'gameplay',
 'as',
 'its',
 'predecessors',
 ',',
 'the',
 'story',
 'runs',
 'parallel',
 'to',
 'the',
 'first',
 'game',
 'and',
 'follows',
 'the']
>>> ' '.join([vocab.idx2word[i] for i in corpus.train[:100]])
'<eos> = Valkyria Chronicles III = <eos> <eos> Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the'
>>> corpus.train
tensor([ 4,  0,  1,  ..., 15,  4,  4])
>>> batchify
<function __main__.batchify(dataset, batch_size=20, device=device(type='cpu'))>
>>> batchify?
>>> for
>>> for i in range(int(len(corpus.train)/batch_size)):
...     print(i)
...
>>> batch_size = 24
>>> for i in range(int(len(corpus.train)/batch_size)):
...     print(i)
...
>>> for i in range(int(len(corpus.train)/batch_size)):
...     print(i*batch_size)
...
>>> for i in range(int(len(corpus.train)/batch_size)):
...     print(i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]])
...
>>> for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     print(i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]])
...
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
... pd.DataFrame(batches_desc)
...
>>> import pandas as pd
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
... pd.DataFrame(batches_desc)
...
     0    1       2
0    0    0   <eos>
1    1   24     the
2    2   48      by
3    3   72  series
4    4   96    game
5    5  120  secret
6    6  144    over
7    7  168    also
8    8  192    both
9    9  216     The
10  10  240    both
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
... pd.DataFrame(batches_desc, columns='batch_num start_tok_num start_tok'.split())
...
    batch_num  start_tok_num start_tok
0           0              0     <eos>
1           1             24       the
2           2             48        by
3           3             72    series
4           4             96      game
5           5            120    secret
6           6            144      over
7           7            168      also
8           8            192      both
9           9            216       The
10         10            240      both
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_num end_tok_pos'.split())
...
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size - 1,
...         vocab.idx2word[corpus.train[i*batch_size - 1]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_num end_tok_pos end_tok'.split())
...
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size - 1,
...         vocab.idx2word[corpus.train[i*batch_size - 1]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_num end_tok_pos'.split())
...
    batch_num  start_tok_pos start_tok  end_tok_num end_tok_pos
0           0              0     <eos>           -1       <eos>
1           1             24       the           23          of
2           2             48        by           47   developed
3           3             72    series           71    Valkyria
4           4             96      game           95       first
5           5            120    secret          119     perform
6           6            144      over          143    carrying
7           7            168      also          167          it
8           8            192      both          191    Sakimoto
9           9            216       The          215           .
10         10            240      both          239          by
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size - 1,
...         vocab.idx2word[corpus.train[i*batch_size - 1]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
...
    batch_num  start_tok_pos start_tok  end_tok_pos    end_tok
0           0              0     <eos>           -1      <eos>
1           1             24       the           23         of
2           2             48        by           47  developed
3           3             72    series           71   Valkyria
4           4             96      game           95      first
5           5            120    secret          119    perform
6           6            144      over          143   carrying
7           7            168      also          167         it
8           8            192      both          191   Sakimoto
9           9            216       The          215          .
10         10            240      both          239         by
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         (i+1)*batch_size - 1,
...         vocab.idx2word[corpus.train[(i+1)*batch_size - 1]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
...
    batch_num  start_tok_pos start_tok  end_tok_pos    end_tok
0           0              0     <eos>           23         of
1           1             24       the           47  developed
2           2             48        by           71   Valkyria
3           3             72    series           95      first
4           4             96      game          119    perform
5           5            120    secret          143   carrying
6           6            144      over          167         it
7           7            168      also          191   Sakimoto
8           8            192      both          215          .
9           9            216       The          239         by
10         10            240      both          263       year
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size + batch_size,
...         vocab.idx2word[corpus.train[i*batch_size + batch_size]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
...
    batch_num  start_tok_pos start_tok  end_tok_pos end_tok
0           0              0     <eos>           24     the
1           1             24       the           48      by
2           2             48        by           72  series
3           3             72    series           96    game
4           4             96      game          120  secret
5           5            120    secret          144    over
6           6            144      over          168    also
7           7            168      also          192    both
8           8            192      both          216     The
9           9            216       The          240    both
10         10            240      both          264       .
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size + batch_size - 1,
...         vocab.idx2word[corpus.train[i*batch_size + batch_size - ]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
...
>>> batches_desc = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     batches_desc.append([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size + batch_size - 1,
...         vocab.idx2word[corpus.train[i*batch_size + batch_size - 1]],
...         ])
... pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
...
    batch_num  start_tok_pos start_tok  end_tok_pos    end_tok
0           0              0     <eos>           23         of
1           1             24       the           47  developed
2           2             48        by           71   Valkyria
3           3             72    series           95      first
4           4             96      game          119    perform
5           5            120    secret          143   carrying
6           6            144      over          167         it
7           7            168      also          191   Sakimoto
8           8            192      both          215          .
9           9            216       The          239         by
10         10            240      both          263       year
>>> batches = []
... for i in range(int(len(corpus.train)/batch_size)):
...     if i > 10:
...         break
...     print([
...         i,
...         i*batch_size,
...         vocab.idx2word[corpus.train[i*batch_size]],
...         i*batch_size + batch_size - 1,
...         vocab.idx2word[corpus.train[i*batch_size + batch_size - 1]],
...         ])
...     batches.append(corpus.train[i*batch_size:i*batch_size + batch_size - 1])
...     print(batches[-1])
... # pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
...
>>> def batchify1(x, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i > num_batches:
...             break
...         print([
...             i,
...             i*batch_size,
...             vocab.idx2word[x[i*batch_size]],
...             i*batch_size + batch_size - 1,
...             vocab.idx2word[x[i*batch_size + batch_size - 1]],
...             ])
...         batches.append(x[i*batch_size:i*batch_size + batch_size - 1])
...     return batches
...
>>> x
>>> x
>>> x
>>> x
>>> def batchify1(x, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i > num_batches:
...             break
...         print([
...             i,
...             i*batch_size,
...             vocab.idx2word[x[i*batch_size]],
...             i*batch_size + batch_size - 1,
...             vocab.idx2word[x[i*batch_size + batch_size - 1]],
...             ])
...         batches.append(x[i*batch_size:i*batch_size + batch_size - 1])
...     return batches
...
>>> batchify1(corpus.train)
[tensor([ 4,  0,  1,  2,  3,  0,  4,  4,  5,  6,  1,  7,  8,  9,  2, 10, 11,  8,
         12, 13, 14, 15,  1]),
 tensor([17, 18,  7, 19, 13, 20, 21, 22, 23,  1,  2,  3, 24, 25, 13, 26, 27, 28,
         29, 30, 31, 32, 33]),
 tensor([35, 36, 37, 38, 39, 17, 40, 41, 15, 42, 43, 44, 45, 43, 25, 13, 46, 26,
         17, 47, 33, 43, 17]),
 tensor([48, 15,  9, 17, 49, 50, 16, 28, 37, 51, 30, 52, 53, 23, 54, 55, 13, 17,
         56, 57, 58, 22, 17]),
 tensor([33, 37, 60, 17, 61, 62, 61, 13, 27, 63, 64, 65, 66, 17, 67, 16, 68, 69,
         17, 70, 71, 72, 73]),
 tensor([75, 76, 77, 37, 78, 79, 80, 17, 81, 65, 61,  9, 82, 61, 15,  4, 83, 33,
         84, 85, 43, 86, 13])]
>>> def batchify1(x, batch_size=8, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i > num_batches:
...             break
...         print([
...             i,
...             i*batch_size,
...             vocab.idx2word[x[i*batch_size]],
...             i*batch_size + batch_size - 1,
...             vocab.idx2word[x[i*batch_size + batch_size - 1]],
...             ])
...         batches.append(x[i*batch_size:i*batch_size + batch_size - 1])
...     return batches
...
>>> batchify1(corpus.train, batch_size=24)
[tensor([ 4,  0,  1,  2,  3,  0,  4,  4,  5,  6,  1,  7,  8,  9,  2, 10, 11,  8,
         12, 13, 14, 15,  1]),
 tensor([17, 18,  7, 19, 13, 20, 21, 22, 23,  1,  2,  3, 24, 25, 13, 26, 27, 28,
         29, 30, 31, 32, 33]),
 tensor([35, 36, 37, 38, 39, 17, 40, 41, 15, 42, 43, 44, 45, 43, 25, 13, 46, 26,
         17, 47, 33, 43, 17]),
 tensor([48, 15,  9, 17, 49, 50, 16, 28, 37, 51, 30, 52, 53, 23, 54, 55, 13, 17,
         56, 57, 58, 22, 17]),
 tensor([33, 37, 60, 17, 61, 62, 61, 13, 27, 63, 64, 65, 66, 17, 67, 16, 68, 69,
         17, 70, 71, 72, 73]),
 tensor([75, 76, 77, 37, 78, 79, 80, 17, 81, 65, 61,  9, 82, 61, 15,  4, 83, 33,
         84, 85, 43, 86, 13])]
>>> batchify1(corpus.train, batch_size=3)
[tensor([4, 0]),
 tensor([2, 3]),
 tensor([4, 4]),
 tensor([6, 1]),
 tensor([8, 9]),
 tensor([10, 11])]
>>> batchify1(corpus.train)
[tensor([4, 0, 1, 2, 3, 0, 4]),
 tensor([5, 6, 1, 7, 8, 9, 2]),
 tensor([11,  8, 12, 13, 14, 15,  1]),
 tensor([17, 18,  7, 19, 13, 20, 21]),
 tensor([23,  1,  2,  3, 24, 25, 13]),
 tensor([27, 28, 29, 30, 31, 32, 33])]
>>> def batchify1(x, batch_size=8, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i > num_batches:
...             break
...         print([
...             i,
...             i*batch_size,
...             vocab.idx2word[x[i*batch_size]],
...             i*batch_size + batch_size - 1,
...             vocab.idx2word[x[i*batch_size + batch_size - 1]],
...             ])
...         batches.append(x[i*batch_size:i*batch_size + batch_size])
...     return batches
...
>>> def batchify1(x, batch_size=8, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i > num_batches:
...             break
...         print([
...             i,
...             i*batch_size,
...             vocab.idx2word[x[i*batch_size]],
...             i*batch_size + batch_size,
...             vocab.idx2word[x[i*batch_size + batch_size]],
...             ])
...         batches.append(x[i*batch_size:i*batch_size + batch_size])
...     return batches
...
>>> batchify1(corpus.train)
[tensor([4, 0, 1, 2, 3, 0, 4, 4]),
 tensor([ 5,  6,  1,  7,  8,  9,  2, 10]),
 tensor([11,  8, 12, 13, 14, 15,  1, 16]),
 tensor([17, 18,  7, 19, 13, 20, 21, 22]),
 tensor([23,  1,  2,  3, 24, 25, 13, 26]),
 tensor([27, 28, 29, 30, 31, 32, 33, 34])]
>>> def batchify1(x, batch_size=8, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i > num_batches:
...             break
...         batches.append(x[i*batch_size:i*batch_size + batch_size])
...     return batches
...
>>> batchify1(corpus.train)
[tensor([4, 0, 1, 2, 3, 0, 4, 4]),
 tensor([ 5,  6,  1,  7,  8,  9,  2, 10]),
 tensor([11,  8, 12, 13, 14, 15,  1, 16]),
 tensor([17, 18,  7, 19, 13, 20, 21, 22]),
 tensor([23,  1,  2,  3, 24, 25, 13, 26]),
 tensor([27, 28, 29, 30, 31, 32, 33, 34])]
>>> batches = batchify1(corpus.train, num_batches=100)
>>> torch.stack(batches)
tensor([[  4,   0,   1,   2,   3,   0,   4,   4],
        [  5,   6,   1,   7,   8,   9,   2,  10],
        [ 11,   8,  12,  13,  14,  15,   1,  16],
        [ 17,  18,   7,  19,  13,  20,  21,  22],
        [ 23,   1,   2,   3,  24,  25,  13,  26],
        [ 27,  28,  29,  30,  31,  32,  33,  34],
        [ 35,  36,  37,  38,  39,  17,  40,  41],
        [ 15,  42,  43,  44,  45,  43,  25,  13],
        [ 46,  26,  17,  47,  33,  43,  17,   1],
        [ 48,  15,   9,  17,  49,  50,  16,  28],
        [ 37,  51,  30,  52,  53,  23,  54,  55],
        [ 13,  17,  56,  57,  58,  22,  17,  59],
        [ 33,  37,  60,  17,  61,  62,  61,  13],
        [ 27,  63,  64,  65,  66,  17,  67,  16],
        [ 68,  69,  17,  70,  71,  72,  73,  74],
        [ 75,  76,  77,  37,  78,  79,  80,  17],
        [ 81,  65,  61,   9,  82,  61,  15,   4],
        [ 83,  33,  84,  85,  43,  86,  13,  87],
        [ 88,  27,  89,  90,  16,  17,  91,  92],
        [ 93,   1,   2,  94,  15,  95,  46,  96],
        [ 17,  97,  98,  16,  17,  48,  13,  46],
        [ 99, 100, 101, 102,  13, 103,  23, 104],
        [ 17,  33, 105,   9,  39,  48, 106,  15],
        [107, 108,   9, 109,  37, 110, 111, 112],
        [113, 114, 115, 116, 117,  13, 118, 119],
        [  1,   2,  94, 120, 121, 122,  15, 123],
        [ 89, 124,  16, 125, 126,  17, 127,  15],
        [ 83,  33, 128, 129, 130, 131, 132,  35],
        [133, 134,  15,   4, 135, 136, 119, 137],
        [138,  43,  25,  13,  37, 131, 139,  35],
        [113,  11,  37, 140, 141,  15, 142, 143],
        [ 13,  46, 144, 145, 146,  13, 118, 119],
        [147, 148, 149,  43, 150,  16, 151, 152],
        [ 15, 135, 131,  99, 153, 154, 155,  37],
        [147, 156,  32, 157,  48,  15, 158,  22],
        [159, 138,  16,   1,   2,  94,  13,   1],
        [  2,   3, 131, 160, 161,  13, 162,  27],
        [163, 164, 165, 119,  17,  33, 128, 148],
        [149, 131, 166,  43, 167,  15,  38, 168],
        [169,  22,  17, 170, 119,  17,  85,  16],
        [  1,   8, 171, 172,  39,  17,  40, 173],
        [ 15,   4,   4,   0,   0, 174,   0,   0],
        [  4,   4, 175, 119, 116,   9,   2, 176],
        [ 13,   1,   2,   3,  26,  27,  28,  29],
        [ 30,  31,  33, 177, 178, 179, 180,  16],
        [ 27,  64,  65,  37, 179, 181,  43, 182],
        [ 80, 183, 184,  15, 185,  78, 186, 187],
        [188, 189,  30, 190, 191, 119, 192, 193],
        [194,  13, 119, 195, 196, 197, 187, 198],
        [199, 200,  37, 197, 187,   9, 201,  15],
        [ 83, 202, 203, 187,  27,  48,  16, 204],
        [182,  13, 205, 206,  23, 207, 151, 208],
        [209, 210,   9, 187,  37, 211,  23, 212],
        [ 78, 206,  15,  83, 213,  22, 214,  56],
        [215,  93,  17, 216, 217, 218,  93, 147],
        [219, 202, 128, 220,   8, 221, 222, 223],
        [ 26, 224,  13,  17, 225,  26, 226, 227],
        [ 22,  17, 202,  15, 228, 182,  13,  17],
        [202, 195, 229,  43,  27, 230,  13, 177],
        [231, 208, 209, 232,  37, 193, 233, 234],
        [ 15, 235,  17, 236,  56, 182,  78, 193],
        [ 30, 237, 238, 182, 239,  22, 240, 241],
        [242,  15, 142,  17,  33, 128, 243,  13],
        [244, 245,  78, 206,  13, 246,  16, 247],
        [248,  27, 249, 250, 251, 252, 253,  43],
        [ 17, 229,  16,  17,  33,  15, 254,  78],
        [ 99, 255, 256, 257, 258,  22,  17,  33],
        [128, 259, 236,   9,  13, 260, 212, 179],
        [ 27, 261, 262,  29,  15,   4,  83,  33],
        [128, 263, 264,  13,  17,   9, 264,  13],
        [ 26, 265,  88, 266, 115,   9,   2,  15],
        [267, 182,  13, 178, 268, 214,  65, 269],
        [ 27, 270,  30, 271, 272,  16,  17, 273],
        [216,   8, 274,  27, 193,  26, 224,  13],
        [ 17, 202, 275,  17, 193, 276,  17, 273],
        [ 43,  47,  30, 277,  15, 123, 193, 208],
        [278, 279, 274, 280,  30, 281,  13, 162],
        [195, 208, 209, 282, 101, 283, 284,  17],
        [285,  16, 225, 195, 286, 283,  15, 287],
        [193, 288,  27, 289,  37, 290,  16, 291],
        [292,  35, 293, 294,   9,  15, 295,  22],
        [296, 195, 208, 209, 297,  22,  27, 298],
        [299,  15, 267,  53,  13, 195, 300, 301],
        [302, 303, 304, 305,  22, 247,  13, 103],
        [ 23, 293, 306, 307,  10, 308,  19, 309],
        [159, 310, 311, 312, 302,  35, 183, 313],
        [ 15, 287, 193, 288, 237,  61, 314,  61],
        [ 13, 315, 316,  22, 214, 193,  15, 317],
        [ 78, 318, 154,  61, 319, 320,  61,  13],
        [321,  78, 322, 315, 151, 323, 324, 325],
        [326, 327,  35,  17,  56,  37, 208, 328],
        [329, 310, 330,  27, 193,  13,  37,  61],
        [331, 314,  61,  13, 321,  78, 332, 333],
        [ 17,  33,  37, 334, 335,   9,  22,  27],
        [193,  15, 336, 337, 331, 314,  13, 214],
        [193, 288,  27, 316,  61, 338, 339,  61],
        [ 13,  27, 340,  30, 341, 342, 343, 151],
        [208, 209, 344,  22, 345,  37, 346, 240],
        [315,  15, 347,  99, 348, 349,   9, 151],
        [335, 247, 350,   9,  93,  17, 273,   8],
        [351, 208, 352,  61, 353, 354,  61,  37]])
>>> torch.stack(batches).size
<function Tensor.size>
>>> torch.stack(batches).size()
torch.Size([101, 8])
>>> def batchify1(x, batch_size=8, num_batches=5):
...     batches = []
...     for i in range(int(len(x)/batch_size)):
...         if i >= num_batches:
...             break
...         batches.append(x[i*batch_size:i*batch_size + batch_size])
...     return torch.stack(batches)
...
>>> batches = batchify1(corpus.train, num_batches=10, batch_size=3)
>>> torch.stack(batches).size()
>>> batches
tensor([[ 4,  0,  1],
        [ 2,  3,  0],
        [ 4,  4,  5],
        [ 6,  1,  7],
        [ 8,  9,  2],
        [10, 11,  8],
        [12, 13, 14],
        [15,  1, 16],
        [17, 18,  7],
        [19, 13, 20]])
>>> batches.size()
torch.Size([10, 3])
>>> batches = batchify1(corpus.train, num_batches=1000, batch_size=10)
>>> batches.size()
torch.Size([1000, 10])
>>> train_epoch
<function __main__.train_epoch(model, train_data, ntokens, criterion=NLLLoss(), lr=2.0)>
>>> train_epoch?
>>> train_epoch(model, batches, ntokens=len(corpus.dictionary.idx2word))
>>> batches = batchify1(corpus.train, num_batches=1000, batch_size=20)
>>> train_epoch(model, batches, ntokens=len(corpus.dictionary.idx2word))
>>> hist -o -p -f hist/ch08_batchify_rnn_input.hist.md
```