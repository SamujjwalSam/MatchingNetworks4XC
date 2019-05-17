# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Class to process and load html files from a directory.

__description__ : Class to process and load html files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : HTMLLoader,

__variables__   :

__methods__     :
"""

from os import listdir,makedirs
from os.path import join,isfile,isdir
import torch.utils.data
from collections import OrderedDict

from file_utils import File_Util
from text_process import clean_wiki,Clean_Text
from logger.logger import logger
from config import configuration as config
from config import platform as plat
from config import username as user


class HTMLLoader(torch.utils.data.Dataset):
    """
    Class to process and load html files from a directory.

    Datasets: Wiki10-31K

    sentences : Wikipedia english texts after parsing and cleaning.
    sentences = {"id1": "wiki_text_1", "id2": "wiki_text_2"}

    classes   : OrderedDict of id to classes.
    classes = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    categories : Dict of class texts.
    categories = {"Computer Science":class_id_1, "Machine Learning":class_id_2}

    samples : {
        "sentences":"",
        "classes":""
        }
    """

    def __init__(self,dataset_name=config["data"]["dataset_name"],
                 data_dir: str = config["paths"]["dataset_dir"][plat][user]):
        """
        Initializes the html loader.

        Args:
            data_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(HTMLLoader,self).__init__()
        self.dataset_name = dataset_name
        # self.data_dir = join(data_dir,self.dataset_name)
        self.data_dir = data_dir
        self.raw_html_dir = join(self.data_dir,dataset_name + "_RawData")
        self.raw_txt_dir = join(self.data_dir,"txt_files")
        logger.debug("Dataset name: [{}], Directory: [{}]".format(self.dataset_name,self.data_dir))
        self.clean = Clean_Text()
        self.sentences,self.classes,self.categories = self.gen_dicts()

    def gen_dicts(self):
        """Filters sentences, classes and categories from wikipedia text.

        :return: Dict of sentences, classes and categories filtered from samples.
        """

        if isdir(self.raw_txt_dir):
            logger.info("Loading data from TXT files.")
            self.samples = self.read_txt_dir(self.raw_txt_dir)
        else:
            logger.info("Could not find TXT files: [{}]".format(self.raw_txt_dir))
            logger.info("Loading data from HTML files.")
            html_parser = self.get_html_parser()
            self.samples = self.read_html_dir(html_parser)

        classes = OrderedDict()
        hid_classes = OrderedDict()
        categories = OrderedDict()
        hid_categories = OrderedDict()
        sentences = OrderedDict()
        cat_idx = 0
        hid_cat_idx = 0
        no_cat_ids = []  # List to store failed parsing cases.
        for doc_id,txt in self.samples.items():
            txt = list(filter(None,txt))  # Removing empty items
            doc,filtered_categories,filtered_hid_categories = self.clean.filter_html_categories_reverse(txt)
            ## assert filtered_categories, "No category information was found for doc_id: [{0}].".format(doc_id)
            if filtered_categories:  ## Check at least one category was successfully filtered from html file.
                sentences[doc_id] = clean_wiki(doc)  ## Removing category information and other texts from html pages.
                for lbl in filtered_categories:
                    if lbl not in categories:  ## If lbl does not exists in categories already, add it and assign a
                        ## new category index.
                        categories[lbl] = cat_idx
                        cat_idx += 1
                    if doc_id in classes:  ## Check if doc_id exists, append if yes.
                        classes[doc_id].append(categories[lbl])
                    else:  ## Create entry for doc_id if does not exist.
                        classes[doc_id] = [categories[lbl]]
            else:  ## If no category was found, store the doc_id in a separate place for later inspection.
                logger.warn("No categories found in document: [{}].".format(doc_id))
                no_cat_ids.append(doc_id)

            ## Shall we use hidden category information?
            if filtered_hid_categories:  ## Check at least one hidden category was successfully filtered from html file.
                for lbl in filtered_hid_categories:
                    if lbl not in hid_categories:  ## If lbl does not exists in hid_categories already, add it and
                        ## assign a new hid_category index.
                        hid_categories[lbl] = hid_cat_idx
                        hid_cat_idx += 1
                    if doc_id in hid_classes:  ## Check if doc_id exists, append if yes.
                        hid_classes[doc_id].append(hid_categories[lbl])
                    else:  ## Create entry for doc_id if does not exist.
                        hid_classes[doc_id] = [hid_categories[lbl]]
        logger.warn("No categories found for: [{}] documents. Storing ids for reference in file '_no_cat_ids'."
                    .format(len(no_cat_ids)))
        File_Util.save_json(hid_classes,self.dataset_name + "_hid_classes",file_path=self.data_dir)
        File_Util.save_json(hid_categories,self.dataset_name + "_hid_categories",file_path=self.data_dir)
        File_Util.save_json(no_cat_ids,self.dataset_name + "_no_cat_ids",file_path=self.data_dir)
        logger.info("Number of sentences: [{}], classes: [{}] and categories: [{}]."
                    .format(len(sentences),len(classes),len(categories)))
        return sentences,classes,categories

    def read_txt_dir(self,raw_txt_dir: str,encoding: str = "iso-8859-1") -> OrderedDict:
        """
        Reads all txt files from [self.raw_txt_dir] folder as str and returns a OrderedDict[str(filename)]=str(content).

        :param raw_txt_dir:
        :param encoding:
        :param html_parser:
        :param data_dir: Path to directory of html files.
        """
        data = OrderedDict()
        if raw_txt_dir is None: raw_txt_dir = self.raw_txt_dir
        logger.info("Raw TXT path: {}".format(raw_txt_dir))
        if isdir(raw_txt_dir):
            for i in listdir(raw_txt_dir):
                if isfile(join(raw_txt_dir,i)) and i.endswith(".txt"):
                    with open(join(raw_txt_dir,i),encoding=encoding) as txt_ptr:
                        data[str(i[:-4])] = str(txt_ptr.read()).splitlines()  ## [:-4] removes the ".txt" from filename.
        return data

    @staticmethod
    def get_html_parser(alt_text: bool = True,ignore_table: bool = True,decode_errors: str = "ignore",
                        default_alt: str = "",
                        ignore_link: bool = True,reference_links: bool = True,bypass_tables: bool = True,
                        ignore_emphasis: bool = True,
                        unicode_snob: bool = False,no_automatic_links: bool = True,no_skip_internal_links: bool = True,
                        single_line_break: bool = True,
                        escape_all: bool = True,ignore_images: object = True):
        """
        Returns a html parser with config, based on: https://github.com/Alir3z4/html2text.

        Usage: https://github.com/Alir3z4/html2text/blob/master/docs/usage.md
        logger.debug(html_parser.handle("<p>Hello, <a href='http://earth.google.com/'>world</a>!"))

        ignore_links    : Ignore converting links from HTML
        images_to_alt   : Discard image data, only keep alt text
        ignore_tables   : Ignore table-related tags (table, th, td, tr) while keeping rows.
        decode_errors   : What to do in case an error is encountered. ignore, strict, replace etc.
        default_image_alt: Inserts the given alt text whenever images are missing alt values.
        :return: html2text parser.
        """
        logger.info("Getting HTML parser.")
        import html2text  ## https://github.com/Alir3z4/html2text

        html_parser = html2text.HTML2Text()
        # html_parser.images_to_alt = alt_text  ## Discard image data, only keep alt text
        html_parser.ignore_tables = ignore_table  ## Ignore table-related tags (table, th, td, tr) while keeping rows.
        html_parser.ignore_images = ignore_images  ## Ignore table-related tags (table, th, td, tr) while keeping rows.
        html_parser.decode_errors = decode_errors  ## Handling decoding error: "ignore", "strict", "replace" etc.
        html_parser.default_image_alt = default_alt  ## Inserts the given alt text whenever images are missing alt values.
        html_parser.ignore_links = ignore_link  ## Ignore converting links from HTML.
        html_parser.reference_links = reference_links  ## Use reference links instead of inline links to create markdown.
        html_parser.bypass_tables = bypass_tables  ## Format tables in HTML rather than Markdown syntax.
        html_parser.ignore_emphasis = ignore_emphasis  ## Ignore all emphasis formatting in the html.
        html_parser.unicode_snob = unicode_snob  ## Use unicode throughout instead of ASCII.
        html_parser.no_automatic_links = no_automatic_links  ## Do not use automatic links like http://google.com.
        ## html_parser.no_skip_internal_links = no_skip_internal_links  ## Turn off skipping of internal links.
        html_parser.skip_internal_links = True
        html_parser.single_line_break = single_line_break  ## Use a single line break after a block element rather than two.
        html_parser.escape_all = escape_all  ## Escape all special characters.
        return html_parser

    def read_html_dir(self,html_parser,encoding="iso-8859-1",specials="""_-@*#'"/\\""",replace=' '):
        """
        Reads all html files in a folder as str and returns a OrderedDict[str(filename)]=str(content).

        :param replace:
        :param specials:
        :param encoding:
        :param html_parser:
        :param data_dir: Path to directory of html files.
        """
        from unidecode import unidecode

        data = OrderedDict()
        # logger.debug("Raw HTML path: {}".format(self.raw_html_dir))
        makedirs(join(self.data_dir,"txt_files"),exist_ok=True)
        if isdir(self.raw_html_dir):
            trans_table = self.clean.make_trans_table(specials=specials,
                                                      replace=replace)  ## Creating mapping to clean sentences.
            for i in listdir(self.raw_html_dir):
                if isfile(join(self.raw_html_dir,i)):
                    with open(join(self.raw_html_dir,i),encoding=encoding) as html_ptr:
                        h_content = html_parser.handle(html_ptr.read())
                        clean_text = unidecode(str(h_content).splitlines()).translate(trans_table)
                        File_Util.write_file(clean_text,i,file_path=join(self.data_dir,"txt_files"))
                        data[str(i)] = clean_text
        return data

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences,self.classes,self.categories

    def get_sentences(self):
        """
        Function to get the entire set of features
        """
        return self.sentences

    def get_classes(self):
        """
        Function to get the entire set of classes.
        """
        return self.classes

    def get_categories(self) -> dict:
        """
        Function to get the entire set of categories
        """
        return self.categories


def main():
    # config = read_config(args)
    doc = """
# Billie Holiday

### From Wikipedia, the free encyclopedia

Jump to: navigation, search

Billie Holiday  
Billie Holiday, 1949  

Billie Holiday, 1949  
Background information  
Birth name Eleanora Fagan  
Also known as Lady Day, Queen of Song  
Born April 7, 1915(1915-04-07)  
Origin Harlem, New York City  
Died July 17, 1959 (aged 44)  
Genre(s) Jazz, vocal jazz, jazz blues, torch songs, ballads, swing  
Occupation(s) Jazz singer, composer  
Instrument(s) Vocals  
Years active 1930s-1959  
Label(s) Columbia (1933-1942, 1958)  
Commodore (1939, 1944)  
Decca (1944-1950)  
Verve (1952-1959)  
MGM (1959)  
Associated acts Ella Fitzgerald, Sarah Vaughan  
Website Billie Holiday Official Site  
Billie Holiday (born Eleanora Fagan; April 7, 1915 â July 17, 1959) was an
American jazz singer and songwriter.

Nicknamed Lady Day[1] by her loyal friend and musical partner Lester Young,
Holiday was a seminal influence on jazz and pop singing. Her vocal style,
strongly inspired by jazz instrumentalists, pioneered a new way of
manipulating phrasing and tempo. Above all, she was admired for her deeply
personal and intimate approach to singing. Critic John Bush wrote that she
"changed the art of American pop vocals forever."[2] She co-wrote only a few
songs, but several of them have become jazz standards, notably "God Bless the
Child", "Don't Explain", and "Lady Sings the Blues". She also became famous
for singing jazz standards written by others, including "Easy Living" and
"Strange Fruit."

## Contents

  * 1 Biography
    * 1.1 Early life
    * 1.2 Early singing career
    * 1.3 The Commodore years and "Strange Fruit"
    * 1.4 Decca Records and "Lover Man"
    * 1.5 Film
    * 1.6 1947 arrest and Carnegie Hall comeback concert
    * 1.7 Early and mid 1950s
    * 1.8 Death
  * 2 Voice
  * 3 References and tributes
  * 4 Songs composed by Holiday
  * 5 Discography
    * 5.1 Select studio albums
    * 5.2 Live recordings
    * 5.3 Box sets
  * 6 Selective awards
    * 6.1 Grammy Hall of Fame
    * 6.2 Grammy Best Historical Album
    * 6.3 Other honors
  * 7 Videography
  * 8 Quote
  * 9 Notes
  * 10 References
  * 11 External links

  
## [edit] Biography

### [edit] Early life

Raised Roman Catholic,[3] Billie Holiday had a difficult childhood, which
greatly affected her life and career. Not much is known about the true details
of her early life, though stories of it appeared in her autobiography, Lady
Sings the Blues, first published in 1956 and later revealed to contain many
inaccuracies.[4]

<IMG>

<IMG>

Billie Holiday at two years old in 1917

Her professional pseudonym was taken from Billie Dove, an actress she admired,
and Clarence Holiday, her probable father. At the outset of her career, she
spelled her last name "Halliday", presumably to distance herself from her
neglectful father, but eventually changed it back to "Holiday".

There is some controversy regarding Holiday's paternity, stemming from a copy
of her birth certificate in the Baltimore archives that lists the father as a
"Frank DeViese". Some historians consider this an anomaly, probably inserted
by a hospital or government worker.[5]

Thrown out of her parents' home in Baltimore after becoming pregnant at
thirteen, Billie's mother, Sadie Fagan, moved to Philadelphia where Billie was
born. Mother and child eventually settled in a poor section of Baltimore. Her
parents married when she was three, but they soon divorced, leaving her to be
raised largely by her mother and other relatives. At the age of 10, she
reported that she had been raped.[6] That claim, combined with her frequent
truancy, resulted in her being sent to The House of the Good Shepherd, a
Catholic reform school, in 1925. It was only through the assistance of a
family friend that she was released two years later.[7] Scarred by these
experiences, Holiday moved to New York City with her mother in 1928. In 1929
Holiday's mother discovered a neighbor, Wilbert Rich, in the act of raping her
daughter; Rich was sentenced to three months in jail.  

### [edit] Early singing career

According to Billie Holiday's own account, she was recruited by a brothel,
worked as a prostitute in 1930, and was eventually imprisoned for a short time
for solicitation. It was in Harlem in the early 1930s that she started singing
for tips in various night clubs. According to legend, penniless and facing
eviction, she sang "Travelin All Alone" in a local club and reduced the
audience to tears. She later worked at various clubs for tips, ultimately
landing at Pod's and Jerry's, a well known Harlem jazz club. Her early work
history is hard to verify, though accounts say she was working at a club named
Monette's in 1933 when she was discovered by talent scout John Hammond.[8]

Hammond arranged for Holiday to make her recording debut in November 1933 with
Benny Goodman singing two songs: "Your Mother's Son-In-Law" and "Riffin' the
Scotch". Goodman was also on hand in 1935, when she continued her recording
career with a group led by pianist Teddy Wilson. Their first collaboration
included "What a Little Moonlight Can Do" and "Miss Brown To You", which
helped to establish Holiday as a major vocalist. She began recording under her
own name a year later, producing a series of extraordinary performances with
groups comprising the Swing Era's finest musicians.

Wilson was signed to Brunswick Records by John Hammond for the purpose of
recording current pop tunes in the new Swing style for the growing jukebox
trade. They were given free rein to improvise the material. Holiday's amazing
method of improvising the melody line to fit the emotion was revolutionary.
(Wilson and Holiday took pedestrian pop tunes like "Twenty-Four Hours A Day"
or "Yankee Doodle Never Went To Town" and turned them into jazz classics with
their arrangements.) With few exceptions, the recordings she made with Wilson
or under her own name during the 1930s and early 1940s are regarded as
important parts of the jazz vocal library.

Billie also wrote songs during the 1930s. Such songs include "Billie's Blues",
"Tell Me More (And Then Some)", "Everything Happens For The Best", "Our Love
Is Different", and "Long Gone Blues".

Among the musicians who accompanied her frequently was tenor saxophonist
Lester Young, who had been a boarder at her mother's house in 1934 and with
whom she had a special rapport. "Well, I think you can hear that on some of
the old records, you know. Some time I'd sit down and listen to 'em myself,
and it sound like two of the same voices, if you don't be careful, you know,
or the same mind, or something like that."[9] Young nicknamed her "Lady Day"
and she, in turn, dubbed him "Prez." She did a three-month residency at Clark
Monroe's Uptown House in New York in 1937. In the late 1930s, she also had
brief stints as a big band vocalist with Count Basie (1937) and Artie Shaw
(1938). The latter association placed her among the first black women to work
with a white orchestra, an arrangement that went against the tenor of the
times.

### [edit] The Commodore years and "Strange Fruit"

Holiday was recording for Columbia in the late 1930s when she was introduced
to "Strange Fruit", a song based on a poem about lynching written by Abel
Meeropol, a Jewish schoolteacher from the Bronx. Meeropol used the pseudonym
"Lewis Allan" for the poem, which was set to music and performed at teachers'
union meetings. It was eventually heard by Barney Josephson, proprietor of
CafÃ© Society, an integrated nightclub in Greenwich Village, who introduced it
to Holiday. She performed it at the club in 1939, with some trepidation,
fearing possible retaliation. Holiday later said that the imagery in "Strange
Fruit" reminded her of her father's death, and that this played a role in her
persistence to perform it. In a 1958 interview, she also bemoaned the fact
that many people did not grasp the song's message: "They'll ask me to 'sing
that sexy song about the people swinging'", she said.[10]

When Holiday's producers at Columbia found the subject matter too sensitive,
Milt Gabler agreed to record it for his Commodore Records. That was done in
April, 1939 and "Strange Fruit" remained in her repertoire for twenty years.
She later recorded it again for Verve. While the Commodore release did not get
airplay, the controversial song sold well, though Gabler attributed that
mostly to the record's other side, "Fine and Mellow", which was a jukebox
hit.[11]

### [edit] Decca Records and "Lover Man"

In addition to owning Commodore Records, Milt Gabler was an A&R man for Decca
Records, and he signed Holiday to the label in 1944. Her first recording for
Decca, "Lover Man", was a song written especially for her by Jimmy Davis,
Roger "Ram" Ramirez, and Jimmy Sherman. Although its lyrics describe a woman
who has never known love ("I long to try something I never had"), its
themeâa woman longing for a missing loverâand its refrain, "Lover man, oh,
where can you be?", struck a chord in wartime America and the record became
one of her biggest hits.

Holiday continued to record for Decca until 1950, including sessions with the
Duke Ellington and Count Basie orchestras, and two duets with Louis Armstrong.
Holiday's Decca recordings featured big bands and, sometimes, strings,
contrasting her intimate small group Columbia accompaniments. Some of the
songs from her Decca repertoire became signatures, including "Don't Explain"
and "Good Morning Heartache".

### [edit] Film

Holiday made one major film appearance, opposite Louis Armstrong in New
Orleans (1947). The musical drama featured Holiday singing with Armstrong and
his band and was directed by Arthur Lubin. Holiday was not pleased that her
role was that of a maid, as she recalled in her autobiography, Lady Sings the
Blues:

> "I thought I was going to play myself in it. I thought I was going to be
Billie Holiday doing a couple of songs in a nightclub setting and that would
be that. I should have known better. When I saw the script, I did. You just
tell one Negro girl who's made movies who didn't play a maid or a whore. I
don't know any. I found out I was going to do a little singing, but I was
still playing the part of a maid."

### [edit] 1947 arrest and Carnegie Hall comeback concert

On May 16, 1947, Holiday was arrested for the possession of narcotics and
drugs in her New York apartment. On May 27, 1947, she was in court. "It was
called 'The United States of America versus Billie Holiday'. And that's just
the way it felt," Holiday recalled in her autobiography, Lady Sings the Blues.
Holiday pleaded guilty and was sentenced to Alderson Federal Prison Camp in
West Virginia. Holiday said she never "sang a note" at Alderson even though
people wanted her to.

Luckily for Holiday, she was released early (March 16, 1948) due to good
behavior. When she arrived at Newark, everybody was there to welcome her back,
including her pianist Bobby Tucker. "I might just as well have wheeled into
Penn Station and had a quiet little get-together with the Associated Press,
United Press, and International News Service."

Ed Fishman (who fought with Joe Glaser to be Holiday's manager) thought of the
idea to throw a comeback concert at Carnegie Hall. Holiday hesitated at the
idea because she thought that nobody would accept her back, but she decided to
go with the idea.

On March 27, 1948, the Carnegie concert was a success. Everything was sold out
before the concert started. It isn't certain how many sets Holiday did. She
did sing Cole Porter's "Night and Day" and "Strange Fruit". The concert was
not recorded.

### [edit] Early and mid 1950s

<IMG>

<IMG>

Billie Holiday, March 23, 1949  
Taken by Carl Van Vechten

Although childless, Billie Holiday had two godchildren: singer Billie Lorraine
Feather, daughter of Leonard Feather, and Bevan Dufty, son of William
Dufty.[12]

Holiday stated that she began using hard drugs in the early 1940s. She married
trombonist Jimmy Monroe on August 25, 1941. While still married to Monroe, she
became romantically involved with trumpeter Joe Guy, her drug dealer,
eventually becoming his common law wife. She finally divorced Monroe in 1947,
and also split with Guy. Because of her 1947 conviction, her New York City
Cabaret Card was revoked which kept her from working in clubs there for the
remaining 12 years of her life, except when she played at the Ebony Club in
1948, where she opened under the permission of John Levy.

By the 1950s, Holiday's drug abuse, drinking, and relations with abusive men
led to deteriorating health. As evidenced by her later recordings, Holiday's
voice coarsened and did not project the vibrance it once had. However, she
retained â and, perhaps, strengthened â the emotional impact of her
delivery (See below).

On March 28, 1952, Holiday married Louis McKay, a Mafia enforcer. McKay, like
most of the men in her life, was abusive, but he did try to get her off drugs.
They were separated at the time of her death, but McKay had plans to start a
chain of Billie Holiday vocal studios, a la Arthur Murray dance schools.

Her late recordings on Verve constitute about a third of her commercial
recorded legacy and are as well remembered as her earlier work for the
Columbia, Commodore and Decca labels. In later years her voice became more
fragile, but it never lost the edge that had always made it so distinctive. On
November 10, 1956, she performed before a packed audience at Carnegie Hall, a
major accomplishment for any artist, especially a black artist of the
segregated period of American history. Her performance of "Fine And Mellow" on
CBS's The Sound of Jazz program is memorable for her interplay with her long-
time friend Lester Young; both were less than two years from death. (see the
clip here)

Holiday first toured Europe in 1954, as part of a Leonard Feather package that
also included Buddy DeFranco and Red Norvo. When she returned, almost five
years later, she made one of her last television appearances for Granada's
"Chelsea at Nine", in London. Her final studio recordings were made for MGM in
1959, with lush backing from Ray Ellis and his Orchestra, who had also
accompanied her on Columbia's Lady in Satin album the previous year â see
below). The MGM sessions were released posthumously on a self-titled album,
later re-titled and re-released as Last Recordings.

Holiday's autobiography, Lady Sings the Blues, was ghostwritten by William
Dufty and published in 1956. Dufty, a New York Post writer and editor then
married to Holiday's close friend Maely Dufty, wrote the book quickly from a
series of conversations with the singer in the Duftys' 93rd Street apartment,
drawing on the work of earlier interviewers as well. His aim was to let
Holiday tell her story her way.[12]

### [edit] Death

On May 31, 1959, she was taken to Metropolitan Hospital in New York suffering
from liver and heart disease. Police officers were stationed at the door to
her room. She was arrested for drug possession as she lay dying and her
hospital room was raided by authorities.[12] Holiday remained under police
guard at the hospital until she died from cirrhosis of the liver on July 17,
1959. In the final years of her life, she had been progressively swindled out
of her earnings, and she died with $0.70 in the bank and $750 (a tabloid fee)
on her person.

## [edit] Voice

<IMG>

This article needs additional citations for verification. Please help improve
this article by adding reliable references (ideally, using inline citations).
Unsourced material may be challenged and removed. (October 2006)  
<IMG>

<IMG>

Billie Holiday photographed by Carl Van Vechten, 1949

Her distinct delivery made Billie Holiday's performances instantly
recognizable throughout her career. Her voice lacked range and was somewhat
thin, plus years of abuse eventually altered the texture of her voice and gave
it a prepossessing fragility. Nonetheless, the emotion with which she imbued
each song remained not only intact but also profound.[13]. Her last major
recording, a 1958 album entitled Lady in Satin, features the backing of a
40-piece orchestra conducted and arranged by Ray Ellis, who said of the album
in 1997:

    I would say that the most emotional moment was her listening to the playback of "I'm a Fool to Want You." There were tears in her eyes ... After we finished the album I went into the control room and listened to all the takes. I must admit I was unhappy with her performance, but I was just listening musically instead of emotionally. It wasn't until I heard the final mix a few weeks later that I realized how great her performance really was.
## [edit] References and tributes

In 1972, Diana Ross portrayed her in the film Lady Sings the Blues, which is
loosely based on the 1959 autobiography of the same name. The 1972 film earned
Ross a nomination for the Academy Award for Best Actress. In 1987, Billie
Holiday was posthumously awarded the Grammy Lifetime Achievement Award. In
1994, the United States Postal Service introduced a Billie Holiday postage
stamp,[14] she ranked #6 on VH1's 100 Greatest Women in Rock n' Roll in 1999,
and she was inducted into the Rock and Roll Hall of Fame in 2000. Over the
years, there have been many tributes to Billie Holiday, including "The Day
Lady Died", a 1959 poem by Frank O'Hara, and "Angel of Harlem", a 1988 release
by the group U2.

## [edit] Songs composed by Holiday

  * "Billie's Blues" (1936)
  * "Don't Explain" (1944)
  * "Everything Happens For The Best" (1939)
  * "Fine and Mellow" (1939)
  * "God Bless the Child" (1941)
  * "Lady Sings the Blues" (1956)
  * "Long Gone Blues" (1939)
  * "Now or Never" (1949)
  * "Our Love Is Different" (1939)
  * "Stormy Blues" (1954)

## [edit] Discography

Holiday recorded extensively for four labels: Columbia Records, issued on its
subsidiary labels Brunswick Records, Vocalion Records, and OKeh Records, from
1933 through 1942, and the label proper in 1958; Commodore Records in 1939 and
1944; Decca Records from 1944 through 1950; and Verve Records, also on its
earlier imprint Clef Records, from 1952 through 1958. Many of Holiday's
recordings appeared on 78 rpm records prior to the long-playing vinyl record
era, and only Clef, Verve, and Columbia issued Holiday albums in the 1950s
during her lifetime that were not compilations of previously released
material. Many compilations have been issued since her death; comprehensive
box sets and a selection of live recordings are listed below.

### [edit] Select studio albums

  * An Evening with Billie Holiday (Clef MGC 144, 1953)
  * Lady Sings the Blues (Verve MGC 721, 1956)
  * Body and Soul (Verve MGV 8197, 1957)
  * Songs for DistinguÃ© Lovers (Verve MGV 8257, 1957)
  * All or Nothing At All (Verve MGV 8329, 1958)
  * Lady in Satin (Columbia CL 1157, 1958)

### [edit] Live recordings

  * Ella Fitzgerald and Billie Holiday at Newport (Verve MGV 8234, 1957)
  * At Monterey 1958 (bootleg BHK 50701, 1988)
  * The Complete 1951 Storyville Club Sessions with Stan Getz (bootleg FSRCD 151, 1991)
  * Lady Day: The Storyville Concerts (Vols. 1 and 2) (Jazz Door 1215, 1991)
  * Summer of '49 (Bandstand 1511, 1998)
  * A Midsummer Night's Jazz at Stratford '57 (bootleg BJH 208, 1999)

### [edit] Box sets

  * The Complete Decca Recordings (GRP 601, 1991)
  * The Complete Billie Holiday on Verve 1945-1959 (Polygram 517658, 1993)
  * The Complete Commodore Recordings (GRP 401, 1997)
  * Lady Day: The Complete Billie Holiday on Columbia 1933â1944 (Columbia Legacy CXK85470, 2001)

## [edit] Selective awards

### [edit] Grammy Hall of Fame

Billie Holiday was posthumously inducted into the Grammy Hall of Fame, which
is a special Grammy award established in 1973 to honor recordings that are at
least twenty-five years old, and that have "qualitative or historical
significance."

Billie Holiday: Grammy Hall of Fame Awards[15]  
Year Recorded Title Genre Label Year Inducted Notes  
1944 "Embraceable You" Jazz (single) Commodore 2005  
1958 Lady in Satin Jazz (album) Columbia 2000  
1945 "Lover Man (Oh, Where Can You Be?)" Jazz (single) Decca 1989  
1939 "Strange Fruit" Jazz (single) Commodore 1978 Listed also in the National
Recording Registry by the Library of Congress in 2002  
1941 "God Bless the Child" Jazz (single) Okeh 1976  
### [edit] Grammy Best Historical Album

The Grammy Award for Best Historical Album has been presented since 1979.

Year Title Label Result  
2002 Lady Day: The Complete Billie Holiday Columbia 1933-1944 Winner  
1994 The Complete Billie Holiday Verve 1945-1959 Winner  
1992 Billie Holiday â The Complete Decca Recordings Verve 1944-1950 Winner  
1980 Billie Holiday â Giants of Jazz Time-Life Winner  
### [edit] Other honors

Year Award Honors Notes  
2004 Ertegun Jazz Hall of Fame[16] Inducted Jazz at Lincoln Center, New York  
2000 Rock and Roll Hall of Fame Inducted Category: "Early Influence"  
1997 ASCAP Jazz Wall of Fame[17] Inducted  
  
1947 Esquire Magazine Gold Award Best Leading Female Vocalist Jazz award  
1946 Esquire Magazine Silver Award Best Leading Female Vocalist Jazz award  
1945 Esquire Magazine Silver Award Best Leading Female Vocalist Jazz award  
1944 Esquire Magazine Gold Award Best Leading Female Vocalist Jazz award  
## [edit] Videography

  * The Emperor Jones, 1933, appeared as an extra
  * Symphony in Black, 1935 short (with Duke Ellington)
  * New Orleans, 1947
  * The Sound of Jazz, CBS Television, December 8, 1957
  * Chelsea at Nine, 1959

## [edit] Quote

"The difficult I can do today. The impossible will take a little longer."[18]

## [edit] Notes

  1. ^ (see "Jazz royalty" regarding similar nicknames)
  2. ^ allmusic ((( Billie Holiday > Biography )))
  3. ^ [1]
  4. ^ Donald Clarke - Wishing On the Moon (2000) pp 12 and 395-9, ISBN 0-306-81136-7
  5. ^ Clarke, Donald. Billie Holiday: Wishing on the Moon. ISBN 0-306-81136-7.
  6. ^ Stuart Nicholson. Billie Holiday. Northeastern University Press. ISBN 1555533035.
  7. ^ Billie Holiday biography at Yahoo.com
  8. ^ "Billie Holiday." Black History Month Biographies. 2004. Gale Group Databases. Mar 1, 2004
  9. ^ 1958 interview with Chris Albertson
  10. ^ Interview with Chris Albertson over WHAT-FM, Philadelphia
  11. ^ Donald Clarke - "Wishing On the Moon" (2000) pp 169
  12. ^ a b c "Billie Holiday's bio, 'Lady Sings the Blues,' may be full of lies, but it gets at jazz great's core". San Francisco Chronicle. http://www.sfgate.com/cgi-bin/article.cgi?f=/c/a/2006/09/18/DDG2VL68691.DTL.
  13. ^ Billie Holiday â a booklet published by New York Jazz Museum in 1970
  14. ^ Billie Holiday postage stamp
  15. ^ Grammy Hall of Fame Database
  16. ^ Ertegun Jazz Hall of Fame 2004
  17. ^ The ASCP Jazz Wall of Fame list
  18. ^ "The James Logan Courier". Billie Holiday. http://www.jameslogancourier.org/index.php?itemid=252. Retrieved on 2006-05-15.

## [edit] References

  * Jack Millar, Fine and Mellow: A Discography of Billie Holiday, 1994, ISBN 1-899161-00-7
  * Julia Blackburn, With Billie, ISBN 0-375-40610-7
  * John Chilton, Billie's Blues: The Billie Holiday Story 1933-1959, ISBN 0-306-80363-1
  * Donald Clarke, Billie Holiday: Wishing on the Moon, ISBN 0-306-81136-7
  * Angela Y. Davis, Blues Legacies and Black Feminism: Gertrude "Ma" Rainey, Bessie Smith, and Billie Holiday, ISBN 0-679-77126-3
  * Leslie Gourse, The Billie Holiday Companion: Seven Decades of Commentary, ISBN 0-02-864613-4
  * Farah Jasmine Griffin, If You Can't Be Free, Be A Mystery: In Search of Billie Holiday, ISBN 0-684-86808-3
  * Billie Holiday with William Dufty, Lady Sings the Blues, ISBN 0-14-006762-0
  * Chris Ingham, Billie Holiday, ISBN 1-56649-170-3
  * Burnett James, Billie Holiday, ISBN 0-946771-05-7
  * Stuart Nicholson, Billie Holiday, ISBN 1-55553-303-5
  * Robert O'Meally, "Lady Day: The Many Faces of Billie Holiday", ISBN 1-55970-147-1

## [edit] External links

Sister project Wikiquote has a collection of quotations related to: Billie
Holiday  
  * [2] Billie Holiday Circle, world's oldest Fan Club for Billie, established 1946.
  * Official Site
  * Official Billie Holiday @ Sony BMG
  * Billie Holiday Discography
  * Complete Billie Holiday Discography
  * Billie Holiday's Music - Pure Vintage
  * Brief biography at Jazz (PBS)
  * Brief biography at American Masters (PBS)
  * The African American Registry - Billie Holiday
  * "Essential Billie Holiday Recordings" by Stuart Nicholson, Jazz.com.
  * Billie Holiday Timeline
  * Billie Holiday's Gravesite
  * Billie Holidayâs Small Band Recordings of 1935-1939
  * Billie Holiday Multimedia Directory

v â¢ d â¢ e

Great American Songbook  
Songwriters

Adair Â· Adams Â· Adler Â· Ager Â· Ahlert Â· Arlen Â· Bacharach Â· Basie Â· A.
Bergman Â· M. Bergman Â· Berlin Â· Bernstein Â· Blake Â· Blane Â· Bloom Â·
Bock Â· Bricusse Â· Brown Â· Burke Â· Cahn Â· Carleton Â· Carmichael Â· Cohan
Â· Coleman Â· Comden Â· H. David Â· M. David Â· Dennis Â· DeRose Â· DeSylva Â·
Dietz Â· Donaldson Â· Dubin Â· Duke Â· Ebb Â· Eliscu Â· Ellington Â· Evans Â·
Fain Â· Fields Â· Freed Â· G. Gershwin Â· I. Gershwin Â· A. Green Â· J. Green
Â· Guettel Â· Hamlisch Â· Hammerstein Â· Harbach Â· Harburg Â· Harnick Â· Hart
Â· Henderson Â· Herman Â· Heyman Â· Jobim Â· Jones Â· Kander Â· Kern Â· Lane
Â· Legrand Â· Leigh Â· Lerner Â· Levant Â· C. Lewis Â· S. Lewis Â· Jay
Livingston Â· Jerry Livingston Â· Loesser Â· Loewe Â· Mancini Â· Mandel Â·
Mann Â· Martin Â· McHugh Â· Mercer Â· Newley Â· Nilsson Â· Noble Â· Parish Â·
Porter Â· A. Previn Â· D. Previn Â· Raksin Â· Raposo Â· Razaf Â· Rodgers Â· D.
Rose Â· V. Rose Â· Ross Â· Schwartz Â· A. Sherman Â· R.B. Sherman Â· R.M.
Sherman Â· Sissle Â· Sondheim Â· Stept Â· Stillman Â· Strayhorn Â· Strouse Â·
Styne Â· Swift Â· Tiomkin Â· Troup Â· Van Heusen Â· Waller Â· Warren Â·
Washington Â· Webb Â· Webster Â· Weill Â· Whiting Â· Wilder Â· Williams Â·
Yellen Â· Youmans Â· Young  
Singers

Alexandria Â· Allison Â· K. Allyson Â· E. Anderson Â· I. Anderson Â· Andrews
Â· J. Andrews Â· Anka Â· Apaka Â· Armstrong Â· Astaire Â· Austin Â· Azama Â·
M. Bailey Â· P. Bailey Â· Baker Â· Barber Â· Bassey Â· Bennett Â· Bergen Â·
Berigan Â· Boone Â· Boswell Â· Bowlly Â· Brewer Â· Brice Â· BublÃ© Â· Calloway
Â· V. Carr Â· Carroll Â· Carter Â· Cassidy Â· Channing Â· Charles Â· Chevalier
Â· Christy Â· Cincotti Â· B. Clark Â· V. Clark Â· Cline Â· Clooney Â· Cole Â·
Columbo Â· Como Â· Connick Â· Connor Â· Cook Â· Cornell Â· Crosby Â· Damone Â·
Dandridge Â· Darin Â· Davis Â· Day Â· Daye Â· Dearie Â· DeShannon Â· Desmond
Â· Dietrich Â· Downey Â· Drake Â· Durante Â· Eberle Â· Eberly Â· Eckstine Â·
Eddy Â· Edwards Â· E. Ennis Â· S. Ennis Â· Etting Â· A. Faye Â· F. Faye Â·
Feinstein Â· Fisher Â· Fitzgerald Â· Flint Â· Ford Â· Forrest Â· Four Freshmen
Â· Franchi Â· Francis Â· Gambarini Â· Garland Â· Gilberto Â· GormÃ© Â· Goulet
Â· Gray Â· Greco Â· Hall Â· Hanshaw Â· Hartman Â· Haymes Â· Hendricks Â·
Herman Â· Hibbler Â· Hildegarde Â· Hilliard Â· Hi-Lo's Â· Ho Â· Holiday Â·
Holman Â· Horn Â· Horne Â· Howard Â· Humes Â· Humperdinck Â· Hunter Â· Hyman
Â· Jackie and Roy Â· James Â· Jefferson Â· Jeffries Â· Jolson Â· A. Jones Â·
E. Jones Â· J. Jones Â· Jordan Â· Kallen Â· Keel Â· Kelly Â· Kenney Â· Kent Â·
Kerr Â· Kiley Â· King Â· Kitt Â· Kral Â· Krall Â· C. Laine Â· F. Laine Â·
Langford Â· Lanza Â· C. Lawrence Â· S. Lawrence Â· Lee Â· Lombardo Â· London
Â· Longet Â· Lucas Â· Lund Â· Lupone Â· Lutcher Â· Lynn Â· Lynne Â· MacDonald
Â· MacRae Â· Maggart Â· D. Martin Â· M. Martin Â· T. Martin Â· Mathis Â·
McCorkle Â· McDonald Â· McRae Â· Mercer Â· Merman Â· Merrill Â· Merry Macs Â·
Midler Â· Mills Â· Minnelli Â· Modernaires Â· Monheit Â· Monro Â· Monroe Â·
Mooney Â· H. Morgan Â· J. Morgan Â· R. Morgan Â· Morse Â· Murphy Â· O'Connell
Â· O'Day Â· O'Hara Â· Page Â· Paris Â· B. Peters Â· Peyroux Â· Piaf Â· Pied
Pipers Â· Pizzarelli Â· Pleasure Â· Prysock Â· Rainey Â· Raitt Â· Raney Â·
Reese Â· Reeves Â· Robeson Â· A. Ross Â· Rushing Â· Russell Â· Scott Â· Shore
Â· Short Â· Simone Â· Simms Â· Sinatra Â· Singers Unlimited Â· Sloane Â· B.
Smith Â· J. Smith Â· Kate Smith Â· Keely Smith Â· Sommers Â· Southern Â·
Stafford Â· Starr Â· Staton Â· Stevens Â· Stewart Â· Streisand Â· Sullivan Â·
Sutton Â· Suzuki Â· Swingle Singers Â· Syms Â· Thornton Â· Tilton Â· Todd Â·
TormÃ© Â· Tracy Â· Tucker Â· Tunnell Â· Umeki Â· Vale Â· VallÃ©e Â· Vaughan Â·
Veloso Â· Wain Â· Ward Â· Warren Â· Warwick Â· Washington Â· Waters Â· Wayne
Â· Whiting Â· Wiley Â· A. Williams Â· J. Williams Â· V. Williams Â· C. Wilson
Â· N. Wilson Â· Wright  
  

Persondata  
NAME Holiday, Billie  
ALTERNATIVE NAMES Fagan, Eleanora; Lady Day  
SHORT DESCRIPTION Jazz singer, composer  
DATE OF BIRTH 1915-4-7  
PLACE OF BIRTH Baltimore, Maryland  
DATE OF DEATH 1959-7-17  
PLACE OF DEATH New York City  
Retrieved from "http://en.wikipedia.org/wiki/Billie_Holiday"

Categories: 1915 births | 1959 deaths | Deaths from cirrhosis | Swing singers
| Torch singers | African American Catholics | Classic female blues singers |
African American singers | American buskers | American female singers |
American jazz singers | English-language singers | Blues Hall of Fame
inductees | Grammy Lifetime Achievement Award winners | Musicians from
Philadelphia, Pennsylvania | People from Baltimore, Maryland | Rock and Roll
Hall of Fame inductees | Women in jazz | Musicians from Maryland | Columbia
Records artists | Decca Records artists | Traditional pop music singers |
Alcohol-related deaths in New York

Hidden categories: Articles needing additional references from October 2006

##### Views

  * Article
  * Discussion
  * Edit this page
  * History

##### Personal tools

  * Log in / create account

##### Navigation

  * Main page
  * Contents
  * Featured content
  * Current events
  * Random article

##### Search



##### Interaction

  * About Wikipedia
  * Community portal
  * Recent changes
  * Contact Wikipedia
  * Donate to Wikipedia
  * Help

##### Toolbox

  * What links here
  * Related changes
  * Upload file
  * Special pages
  * Printable version
  * Permanent link
  * Cite this page

##### Languages

  * Ø§ÙØ¹Ø±Ø¨ÙØ©
  * Bosanski
  * ÐÑÐ»Ð³Ð°ÑÑÐºÐ¸
  * CatalÃ 
  * Äesky
  * Dansk
  * Deutsch
  * Eesti
  * ÎÎ»Î»Î·Î½Î¹ÎºÎ¬
  * EspaÃ±ol
  * Esperanto
  * ÙØ§Ø±Ø³Û
  * FranÃ§ais
  * Galego
  * Hrvatski
  * Ido
  * Bahasa Indonesia
  * Italiano
  * ×¢××¨××ª
  * á¥áá áá£áá
  * Kiswahili
  * Latina
  * LietuviÅ³
  * Magyar
  * Nederlands
  * æ¥æ¬èª
  * âªNorsk (bokmÃ¥l)â¬
  * Occitan
  * Polski
  * PortuguÃªs
  * RomÃ¢nÄ
  * Ð ÑÑÑÐºÐ¸Ð¹
  * Sicilianu
  * Suomi
  * Svenska
  * Tagalog
  * à¹à¸à¸¢
  * TÃ¼rkÃ§e
  * Ð£ÐºÑÐ°ÑÐ½ÑÑÐºÐ°
  * ç²µèª
  * ä¸­æ

Powered by MediaWiki

Wikimedia Foundation

  * This page was last modified on 4 April 2009, at 00:21.
  * All text is available under the terms of the GNU Free Documentation License. (See Copyrights for details.)   
Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a U.S.
registered 501(c)(3) tax-deductible nonprofit charity.  

  * Privacy policy
  * About Wikipedia
  * Disclaimers


""".splitlines()
    cls = HTMLLoader()
    returned = cls.filter_html_categories(doc)

    sentences_val,classes_val,categories_val = cls.get_data()
    logger.print_dict(sentences_val)
    logger.print_dict(classes_val)
    logger.print_dict(categories_val)


if __name__ == '__main__':
    main()
