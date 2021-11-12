export const models = [
  {
    key: 'knn',
    params: [
      {
        key: 'n',
        label: 'Neighbors',
        type: 'number',
        default: 5,
      },
      {
        key: 'category',
        label: 'Movie category',
        type: 'select',
        options: ['comedy', 'horror'],
      },
      {
        key: 'exclude',
        label: 'Exclude history',
        type: 'bool',
        default: true,
      },
      {
        key: 'normalize',
        label: 'Normalize distances',
        type: 'bool',
        default: false,
      },
    ],
  },
  {
    key: 'vasp',
    params: [
      {
        key: 'h',
        label: 'Some parameter',
        type: 'text',
      },
    ],
  },
];

export const metrics = {
  knn: {
    recall20: 0.4,
    recall50: 0.23,
    ndcg100: 0.54,
  },
  vasp: {
    recall20: 0.34,
    recall50: 0.14,
    ndcg100: 0.99,
  },
};

export const movies = [
  {
    title: 'Tom and Huck (1995)',
    description:
      'Two best friends witness a murder and embark on a series of adventures in order to prove the innocence of the man wrongly accused of the crime.',
    image:
      'https://m.media-amazon.com/images/M/MV5BN2ZkZTMxOTAtMzg1Mi00M2U0LWE2NWItZDg4YmQyZjVkMDdhXkEyXkFqcGdeQXVyNTM5NzI0NDY@..jpg',
    subtitle: 'Adventure, Comedy, Drama',
    caption: 'English',
    id: 6000,
  },
  {
    title: 'Dracula: Dead and Loving It (1995)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BZWQ0ZDFmYzMtZGMyMi00NmYxLWE0MGYtYzM2ZGNhMTE1NTczL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMjM5ODMxODc@..jpg',
    subtitle: 'Comedy, Fantasy, Horror',
    caption: 'English, German',
    id: 3380,
  },
  {
    title: 'Cutthroat Island (1995)',
    description:
      'A female pirate and her companion race against their rivals to find a hidden island that contains a fabulous treasure.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMDg2YTI0YmQtYzgwMi00Zjk4LWJkZjgtYjg0ZDE2ODUzY2RlL2ltYWdlXkEyXkFqcGdeQXVyNjQzNDI3NzY@..jpg',
    subtitle: 'Action, Adventure, Comedy',
    caption: 'English',
    id: 740,
  },
  {
    title: 'Sense and Sensibility (1995)',
    description:
      'Rich Mr. Dashwood dies, leaving his second wife and her three daughters poor by the rules of inheritance. The two eldest daughters are the title opposites.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNzk1MjU3MDQyMl5BMl5BanBnXkFtZTcwNjc1OTM2MQ@@..jpg',
    subtitle: 'Drama, Romance',
    caption: 'English, French',
    id: 391,
  },
  {
    title: 'Now and Then (1995)',
    description:
      'Four 12-year-old girls grow up together during an eventful small-town summer in 1970.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM2MDQ1YjUtMGM0NC00NmFlLTljMDktZjJiNWRhMWYxOWYyXkEyXkFqcGdeQXVyNjgzMjI4ODE@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'English',
    id: 2874,
  },
  {
    title: "Mr. Holland's Opus (1995)",
    description: 'A frustrated composer finds fulfillment as a high school music teacher.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDZhNDRlZjAtYzdhNy00ZjU1LWFlMDYtNjA5NjliM2Y5ZmVjL2ltYWdlXkEyXkFqcGdeQXVyNjE5MjUyOTM@..jpg',
    subtitle: 'Drama, Music',
    caption: 'English, American Sign Language',
    id: 89,
  },
  {
    title: 'Kicking and Screaming (1995)',
    description:
      'A bunch of guys hang around their college for months after graduation, continuing a life much like the one before graduation.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNWU2YjdlN2ItNTk2OS00MzMwLTlhYjctNDI0MDI0NTQ3OWY0XkEyXkFqcGdeQXVyNzI1NzMxNzM@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'English',
    id: 5235,
  },
  {
    title: 'Angels and Insects (1995)',
    description: 'In the 1800s a naturalist marries into a family of British country gentry.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZTc1MzY1ODAtMDhlMS00NjgyLTlkNTEtZTUwYTM4MzFkNWNmXkEyXkFqcGdeQXVyNDE5MTU2MDE@..jpg',
    subtitle: 'Drama, Romance',
    caption: 'English',
    id: 1025,
  },
  {
    title: 'Muppet Treasure Island (1996)',
    description: "The Muppets' twist on the classic tale.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMTlmNzhiMWEtOWVjZC00NmM0LTgxNDItMDJmYTkxYTZkY2FjXkEyXkFqcGdeQXVyNTUyMzE4Mzg@..jpg',
    subtitle: 'Action, Adventure, Comedy',
    caption: 'English',
    id: 907,
  },
  {
    title: 'Before and After (1996)',
    description:
      'Two parents deal with the effects when their son is accused of murdering his girlfriend.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOWJmODIwYWUtNTNkYy00Njk4LWJkODktMzA5ZTdkMTZhZmZiXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    subtitle: 'Crime, Drama, Mystery',
    caption: 'English',
    id: 3648,
  },
  {
    title: "Young Poisoner's Handbook, The (1995)",
    description:
      'This film is based on a true story about a British teenager who allegedly poisoned family, friends, and co-workers. Graham is highly intelligent, but completely amoral. He becomes ...',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTg0MTE5OTMzNV5BMl5BanBnXkFtZTcwMDM0MzkzMQ@@..jpg',
    subtitle: 'Crime, Drama',
    caption: 'English',
    id: 4028,
  },
  {
    title: 'Up Close and Personal (1996)',
    description:
      'An ambitious young woman, determined to build a career in television journalism, gets good advice from her first boss, and they fall in love.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZTZjYmNkZTYtOTA0Zi00NDcyLWI2YzQtOTgyZjQ2YzM5Y2E2XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Drama, Romance',
    caption: 'English',
    id: 1988,
  },
  {
    title: 'Amazing Panda Adventure, The (1995)',
    description:
      'A young American boy visiting in China helps his zoologist father rescue a panda cub from unscrupulous poachers while his panda reserve is threatened with closure from officious bureaucrats.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZmUyZWJmZDktMjE0Yy00MjUwLTlmMmUtNTk0YjFkODUxZTVkXkEyXkFqcGdeQXVyNzc5MjA3OA@@..jpg',
    subtitle: 'Adventure, Drama, Family',
    caption: 'English',
    id: 2814,
  },
  {
    title: 'Moonlight and Valentino (1995)',
    description:
      'A young widow still grieving over the death of her husband finds herself being comforted by a local housepainter.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYjBhZDFhYWMtYTA0MS00NTZkLWI2YTAtMGJmZmVlNmE3YmE1XkEyXkFqcGdeQXVyMTA0MjU0Ng@@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'English',
    id: 7195,
  },
  {
    title: 'Death and the Maiden (1994)',
    description:
      'A political activist is convinced that her guest is a man who once tortured her for the government.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTgzZTcwNDItYjhhMy00NmViLWFjZjEtZGQ5YWQxYzk0OWU4XkEyXkFqcGdeQXVyMjgxMzgyNjI@..jpg',
    subtitle: 'Drama, Mystery, Thriller',
    caption: 'English',
    id: 1148,
  },
  {
    title: 'Dumb & Dumber (Dumb and Dumber) (1994)',
    description:
      'After a woman leaves a briefcase at the airport terminal, a dumb limo driver and his dumber friend set out on a hilarious cross-country road trip to Aspen to return it.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDQwMjNiMTQtY2UwYy00NjhiLTk0ZWEtZWM5ZWMzNGFjNTVkXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Comedy',
    caption: 'English, Swedish, German',
    id: 715,
  },
  {
    title: 'Pushing Hands (Tui shou) (1992)',
    description: 'All the while, Master Chu tries to find his place in the foreign American world.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMmM1ODExNjctMTg4YS00NTg1LWI5OGEtZjExMjFlMWZmN2VhXkEyXkFqcGdeQXVyNjU1MDM2NjY@..jpg',
    subtitle: 'Comedy, Drama',
    caption: 'Mandarin, English',
    id: 7576,
  },
  {
    title: 'Quick and the Dead, The (1995)',
    description:
      "A female gunfighter returns to a frontier town where a dueling tournament is being held, which she enters in an effort to avenge her father's death.",
    image:
      'https://m.media-amazon.com/images/M/MV5BOTI2ZTZmMmItMmM3YS00ZjUwLWJiODMtMmRjMWM4NDE0OWFhXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Action, Romance, Thriller',
    caption: 'English, Spanish',
    id: 1228,
  },
  {
    title: 'Strawberry and Chocolate (Fresa y chocolate) (1993)',
    description:
      'This Oscar nominated film is the story of two men who are opposites, one gay, the other straight, one a fierce communist, the other a fierce individualist, one suspicious, the other accepting, and how they come to love each other.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMzhjMDQ1YTctNTNmNS00NDZjLWFjYjUtM2FhZTA2NmYxZjJlXkEyXkFqcGdeQXVyMTA0MjU0Ng@@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'Spanish, English, French',
    id: 1927,
  },
  {
    title: 'Clear and Present Danger (1994)',
    description:
      'CIA Analyst Jack Ryan is drawn into an illegal war fought by the US government against a Colombian drug cartel.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNDczOWNiMmEtZjA4MS00NDMzLWExNTktYjc0MGU0YTQ3ZDExXkEyXkFqcGdeQXVyNjU0OTQ0OTY@..jpg',
    subtitle: 'Action, Crime, Drama',
    caption: 'English',
    id: 356,
  },
  {
    title: 'Four Weddings and a Funeral (1994)',
    description:
      'Over the course of five social occasions, a committed bachelor must consider the notion that he may have discovered love.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTMyNzg2NzgxNV5BMl5BanBnXkFtZTcwMTcxNzczNA@@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'English, British Sign Language',
    id: 357,
  },
  {
    title: 'Mrs. Parker and the Vicious Circle (1994)',
    description:
      'Dorothy Parker remembers the heyday of the Algonquin Round Table, a circle of friends whose barbed wit, like hers, was fueled by alcohol and flirted with despair.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNWUwMDdhZWMtOGZkYy00ZTg2LTgwNTktZDVhZDM1MmFhODI4XkEyXkFqcGdeQXVyMjA0MzYwMDY@..jpg',
    subtitle: 'Biography, Drama',
    caption: 'English',
    id: 1094,
  },
  {
    title: 'Frank and Ollie (1995)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMzc5M2NkNTYtNDg3NS00NTQ1LWJjMzYtZjhmMGI1NzkzNGY0XkEyXkFqcGdeQXVyMTQ3Njg3MQ@@..jpg',
    subtitle: null,
    caption: null,
    id: 10914,
  },
  {
    title: 'Highlander III: The Sorcerer (a.k.a. Highlander: The Final Dimension) (1994)',
    description:
      'Deceived that he had won the Prize, Connor MacLeod awakens from a peaceful life when an entombed immortal magician comes seeking the Highlander.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNzZmNGNmMzgtMjA2YS00YzY3LThmMjYtOTVmMzg4ZGY2YWJjXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Action, Fantasy, Romance',
    caption: 'English',
    id: 2304,
  },
  {
    title: 'Cops and Robbersons (1994)',
    description:
      'A counterfeiter with a habit of "eliminating" the competition moves in next door to the Robbersons. Two cops move in with the Robbersons for a stakeout.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMjcxM2VkZDEtYmExZi00ODRhLWI3NGItNjZiM2IxOGQxODM5XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Comedy, Crime, Thriller',
    caption: 'English',
    id: 2309,
  },
  {
    title: 'Dazed and Confused (1993)',
    description:
      'The adventures of high school and junior high students on the last day of school in May 1976.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM5MDY5MDQyOV5BMl5BanBnXkFtZTgwMzM3NzMxMDE@..jpg',
    subtitle: 'Comedy',
    caption: 'English',
    id: 1016,
  },
  {
    title: 'Flesh and Bone (1993)',
    description:
      "Decades later, a son of a killer falls in love with a girl, whose family's horrifying murder he saw in childhood.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMzEzNTA3N2MtNzEyNC00ZDA3LTkwNGMtYWEyODRlNjZhZDQ0XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Drama, Mystery, Romance',
    caption: 'English',
    id: 6245,
  },
  {
    title: 'Orlando (1992)',
    description:
      'After Queen Elizabeth I commands him not to grow old, a young nobleman struggles with love and his place in the world.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYmY1OTA3MjAtYjQxOC00OTlkLWExZWQtMjc3ZjExOWFhM2UwXkEyXkFqcGdeQXVyMTA0MjU0Ng@@..jpg',
    subtitle: 'Biography, Drama, Fantasy',
    caption: 'English, French',
    id: 1972,
  },
  {
    title: 'Radioland Murders (1994)',
    description: 'A series of mysterious crimes confuses existence of a radio network.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMWRkYzc3YjYtNWY3Yy00Y2FmLTgyOTYtM2U2ZDg0MDA0MTIwXkEyXkFqcGdeQXVyNDk3NzU2MTQ@..jpg',
    subtitle: 'Comedy, Crime, Drama',
    caption: 'English',
    id: 6727,
  },
  {
    title: 'Shadowlands (1993)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTE2MGEzMDctZTZlMi00MjY1LWI5NmQtYmJlZGJiYjkwNWQ5XkEyXkFqcGdeQXVyMTMxMTY0OTQ@..jpg',
    subtitle: 'Biography, Drama, Romance',
    caption: 'English',
    id: 397,
  },
  {
    title: 'Tough and Deadly (1995)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BZGU5ZDkxZWEtOWI0My00NGU4LTk4Y2EtOTFkNDcwMTNhZmM2XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: null,
    caption: null,
    id: 11166,
  },
  {
    title: 'Snow White and the Seven Dwarfs (1937)',
    description:
      'Exiled into the dangerous forest by her wicked stepmother, a princess is rescued by seven dwarf miners who make her part of their household.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTQwMzE2Mzc4M15BMl5BanBnXkFtZTcwMTE4NTc1Nw@@..jpg',
    subtitle: 'Animation, Family, Fantasy',
    caption: 'English',
    id: 257,
  },
  {
    title: 'Beauty and the Beast (1991)',
    description:
      "A prince cursed to spend his days as a hideous monster sets out to regain his humanity by earning a young woman's love.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMzE5MDM1NDktY2I0OC00YWI5LTk2NzUtYjczNDczOWQxYjM0XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Animation, Family, Fantasy',
    caption: 'English, French',
    id: 258,
  },
  {
    title: 'Love and a .45 (1994)',
    description:
      'A small time crook flees to Mexico to evade the authorities, loan sharks, and his murderous ex-partner with only his fiancé and a trusted Colt .45.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYTQ1YzE3NzUtOTZmNC00NzE0LWJiZmUtMjhkMjU1MmVhNTZmXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    subtitle: 'Crime, Romance, Thriller',
    caption: 'English',
    id: 8893,
  },
  {
    title: 'Candyman: Farewell to the Flesh (1995)',
    description:
      'The Candyman arrives in New Orleans and sets his sights on a young woman whose family was ruined by the immortal killer years before.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDA2ZWE2YTctN2JiNi00NjdmLThhYWQtN2JjY2M4MTNhM2I5XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Horror, Thriller',
    caption: 'English',
    id: 7580,
  },
  {
    title: 'Bread and Chocolate (Pane e cioccolata) (1973)',
    description:
      'Italian immigrant Nino steadfastly tries to become a member of Swiss Society no matter how awful his situation becomes.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMWFjYjNiZjItMjU3Yy00YTk3LWFhZmMtN2NlZTZjYTE1YTIzXkEyXkFqcGdeQXVyNjg3MTIwODI@..jpg',
    subtitle: 'Comedy, Drama',
    caption: 'Italian, German, English, French, Greek, Spanish',
    id: 4183,
  },
  {
    title: 'Thin Line Between Love and Hate, A (1996)',
    description:
      "An observable, fast-talking party man Darnell Wright, gets his punishment when one of his conquests takes it personally and comes back for revenge in this 'Fatal Attraction'-esque comic ...",
    image:
      'https://m.media-amazon.com/images/M/MV5BZTAwOWRlYzMtZjM0Yy00NTU3LWFkMzAtOTVlMDQ4NDVkZTUxXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    subtitle: 'Comedy, Crime, Drama',
    caption: 'English',
    id: 7581,
  },
  {
    title: 'Land and Freedom (Tierra y libertad) (1995)',
    description:
      'David is an unemployed communist that comes to Spain in 1937 during the civil war to enroll the republicans and defend the democracy against the fascists. He makes friends between the soldiers.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMmE2ZmE0ZDUtYzljOS00ZWZmLWIyNmYtNzkwNGM4MGNlMWY3L2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTYxNjkxOQ@@..jpg',
    subtitle: 'Drama, War',
    caption: 'English, Spanish, Catalan',
    id: 1973,
  },
  {
    title: 'Jack and Sarah (1995)',
    description: 'A young American woman becomes a nanny in the home of a recent British widower.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNTI0YzJjNWQtZjAwZi00OTNmLThiNDgtODY2M2QxZGI4NTAwXkEyXkFqcGdeQXVyMTczNjQwOTY@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'English',
    id: 2322,
  },
  {
    title: 'Moll Flanders (1996)',
    description:
      'The daughter of a thief, young Moll is placed in the care of a nunnery after the execution of her mother. However, the actions of an abusive Priest lead Moll to rebel as a teenager, ...',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDZlODcwNGUtN2JhNi00Njc2LWJmZjYtNWI2ZWE1MTUyNDFjXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    subtitle: 'Drama, Romance',
    caption: 'English',
    id: 1941,
  },
  {
    title: 'James and the Giant Peach (1996)',
    description:
      'An orphan who lives with his two cruel aunts befriends anthropomorphic bugs who live inside a giant peach, and they embark on a journey to New York City.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTNkNWIwNGUtNTJlOC00NDU3LTk0NWEtNjNjNDM4NzRiNThkXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Animation, Adventure, Family',
    caption: 'English',
    id: 1382,
  },
  {
    title: 'Kids in the Hall: Brain Candy (1996)',
    description:
      "A pharmaceutical scientist creates a pill that makes people remember their happiest memory, and although it's successful, it has unfortunate side effects.",
    image:
      'https://m.media-amazon.com/images/M/MV5BZDJlZDE4N2UtMjhjMC00ZWYyLWFiNjItZjI1NjY0MDE3YjE4XkEyXkFqcGdeQXVyNzc5MjA3OA@@..jpg',
    subtitle: 'Comedy',
    caption: 'English',
    id: 3300,
  },
  {
    title: 'Mulholland Falls (1996)',
    description:
      "In 1950's Los Angeles, a special crime squad of the LAPD investigates the murder of a young woman.",
    image:
      'https://m.media-amazon.com/images/M/MV5BOTcwMDE3MTQ5OV5BMl5BanBnXkFtZTcwMzA0MDQ1NA@@..jpg',
    subtitle: 'Crime, Drama, Mystery',
    caption: 'English',
    id: 1942,
  },
  {
    title: 'Of Love and Shadows (1994)',
    description:
      "Chile 1973 is ruled by the dictator Pinochet. The wealthy don't see the violence, terror, executions etc. including Irene. She's engaged to an officer in the fascist military. She meets Francisco who opens her eyes to truth and love.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMTIwOTMxMjk4OV5BMl5BanBnXkFtZTcwMzUzODkyMQ@@..jpg',
    subtitle: 'Drama, Romance, Thriller',
    caption: 'English',
    id: 7289,
  },
  {
    title: 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
    description:
      'An insane general triggers a path to nuclear holocaust that a War Room full of politicians and generals frantically tries to stop.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZWI3ZTMxNjctMjdlNS00NmUwLWFiM2YtZDUyY2I3N2MxYTE0XkEyXkFqcGdeQXVyNzkwMjQ5NzM@..jpg',
    subtitle: 'Comedy',
    caption: 'English, Russian',
    id: 276,
  },
  {
    title: 'Carmen Miranda: Bananas Is My Business (1994)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM4ODQyMzM5MV5BMl5BanBnXkFtZTcwNDYzMjYxMQ@@..jpg',
    subtitle: null,
    caption: null,
    id: 10816,
  },
  {
    title: 'Marlene Dietrich: Shadow and Light (1996)',
    description: null,
    image: 'Error',
    subtitle: null,
    caption: null,
    id: 9842,
  },
  {
    title: 'Last Klezmer: Leopold Kozlowski, His Life and Music, The (1994)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTk2NjAwNTM1Ml5BMl5BanBnXkFtZTcwNDEwNDkyMQ@@..jpg',
    subtitle: null,
    caption: null,
    id: 7586,
  },
  {
    title: "My Life and Times With Antonin Artaud (En compagnie d'Antonin Artaud) (1993)",
    description:
      'May, 1946, in Paris young poet Jacques Prevel meets Antonin Artaud, the actor, artist and writer just released from a mental asylum. Over ten months, we follow the mad Artaud from his cruel...',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM2MjE5MTQ5MF5BMl5BanBnXkFtZTcwOTMzNzcyMQ@@..jpg',
    subtitle: 'Biography, Drama',
    caption: 'French',
    id: 5879,
  },
  {
    title: 'Walking and Talking (1996)',
    description:
      "Just as Amelia thinks she's over her anxiety and insecurity, her best friend announces her engagement, bringing her anxiety and insecurity right back.",
    image:
      'https://m.media-amazon.com/images/M/MV5BODVjODIyNDYtZTU2Ny00MDIxLTg1ZTEtYjJlNDkyNDEzNDcwXkEyXkFqcGdeQXVyMTMxMTY0OTQ@..jpg',
    subtitle: 'Comedy, Drama, Romance',
    caption: 'English',
    id: 5151,
  },
  {
    title: 'Lotto Land (1995)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTkwMDUxMTQyNl5BMl5BanBnXkFtZTgwMzIwNDgwMzE@..jpg',
    subtitle: null,
    caption: null,
    id: 10167,
  },
  {
    title: 'Crows and Sparrows (Wuya yu maque) (1949)',
    description:
      'A story of a corrupt party official who attempts to sell an apartment building he has appropriated from the original owner and the struggles of the tenants to prevent themselves being thrown onto the street.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMWUyYTk0MjItZmZmZS00NmI3LTk5NzUtYjJkYTBiNzNmMDZmXkEyXkFqcGdeQXVyNzI1NzMxNzM@..jpg',
    subtitle: 'Drama',
    caption: 'Mandarin',
    id: 10171,
  },
  {
    title: 'Island of Dr. Moreau, The (1996)',
    description:
      'After being rescued and brought to an island, a man discovers that its inhabitants are experimental animals being turned into strange-looking humans, all of it the work of a visionary doctor.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYjQxNmM5ODItMGM3Yi00N2VlLTkwNTEtNDZkNzUyYzA3MmIxXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Horror, Sci-Fi, Thriller',
    caption: 'English, Indonesian',
    id: 1904,
  },
  {
    title: 'Land Before Time III: The Time of the Great Giving (1995)',
    description: null,
    image: 'https://m.media-amazon.com/images/M/MV5BMTI1OTU3NjkwMl5BMl5BanBnXkFtZTYwOTU2MDg5..jpg',
    subtitle: null,
    caption: null,
    id: 5223,
  },
  {
    title: 'Band Wagon, The (1953)',
    description:
      'A pretentiously artistic director is hired for a new Broadway musical and changes it beyond recognition.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNGUxZmJkZTgtMmI1Ny00Mzg3LWFlY2QtMjNkM2QwZGVhYTQyL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyNjc1NTYyMjg@..jpg',
    subtitle: 'Comedy, Musical, Romance',
    caption: 'English, French, German',
    id: 3557,
  },
  {
    title: 'Ghost and Mrs. Muir, The (1947)',
    description:
      'In 1900, a young widow finds her seaside cottage is haunted and forms a unique relationship with the ghost.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNzYxMjIyMmYtMDI2OS00YTgyLWExZWMtODdkZTk1NzRlODgzXkEyXkFqcGdeQXVyMTAwMzUyOTc@..jpg',
    subtitle: 'Comedy, Drama, Fantasy',
    caption: 'English',
    id: 2338,
  },
  {
    title: 'Angel and the Badman (1947)',
    description:
      'Quirt Evans, an all round bad guy, is nursed back to health and sought after by Penelope Worth, a Quaker girl. He eventually finds himself having to choose between his world and the world Penelope lives in.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYWI1NzdhYTMtZGE2ZS00ZjJjLThmZWQtNjY4ODcxNWM2YjQ3XkEyXkFqcGdeQXVyMTI1NDQ4NQ@@..jpg',
    subtitle: 'Romance, Western',
    caption: 'English',
    id: 5420,
  },
  {
    title: 'Last Man Standing (1996)',
    description:
      'A drifting gunslinger-for-hire finds himself in the middle of an ongoing war between the Irish and Italian mafia in a Prohibition era ghost town.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOThkYmJjYTMtOWMzNC00ZjQ4LWI4NzAtYjRlMDA3ZWMyYWRmXkEyXkFqcGdeQXVyNDIyMjczNjI@..jpg',
    subtitle: 'Action, Crime, Drama',
    caption: 'English, Spanish',
    id: 2798,
  },
  {
    title: 'Winnie the Pooh and the Blustery Day (1968)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BODgyNjM3NWUtZDAxMi00NzRhLWE0NmMtMjIxYTY1MWJlZTQyXkEyXkFqcGdeQXVyNzY1NDgwNjQ@..jpg',
    subtitle: null,
    caption: null,
    id: 406,
  },
  {
    title: 'Bedknobs and Broomsticks (1971)',
    description:
      'An apprentice witch, three kids and a cynical magician conman search for the missing component to a magic spell to be used in the defense of Britain in World War II.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTUxMTY3MTE5OF5BMl5BanBnXkFtZTgwNTQ0ODgxMzE@..jpg',
    subtitle: 'Animation, Adventure, Comedy',
    caption: 'English, German',
    id: 4781,
  },
  {
    title: 'Alice in Wonderland (1951)',
    description:
      'Alice stumbles into the world of Wonderland. Will she get home? Not if the Queen of Hearts has her way.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTgyMjM2NTAxMF5BMl5BanBnXkFtZTgwNjU1NDc2MTE@..jpg',
    subtitle: 'Animation, Adventure, Family',
    caption: 'English',
    id: 1917,
  },
  {
    title: 'Fox and the Hound, The (1981)',
    description:
      'A little fox named Tod, and Copper, a hound puppy, vow to be best buddies forever. But as Copper grows into a hunting dog, their unlikely friendship faces the ultimate test.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMDQzZWRlYjctMjE0ZS00NWU5LWE3NzktYzZiNzJiM2UxMDczXkEyXkFqcGdeQXVyNjUwNzk3NDc@..jpg',
    subtitle: 'Animation, Adventure, Drama',
    caption: 'English',
    id: 1907,
  },
  {
    title: 'Ghost and the Darkness, The (1996)',
    description:
      'A bridge engineer and an experienced old hunter begin a hunt for two lions after they start attacking local construction workers.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNWQ4NDRiMWItNGI5Yi00N2U1LTlkMGQtM2VjMzdkZTU0YzYyXkEyXkFqcGdeQXVyNTc1NTQxODI@..jpg',
    subtitle: 'Adventure, Drama, Thriller',
    caption: 'English, Hindi',
    id: 2152,
  },
  {
    title: 'Aladdin and the King of Thieves (1996)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BODFkMjE5YzAtMDFkOC00ZDNhLTkwNmQtODk1Y2VhNThlNWJhXkEyXkFqcGdeQXVyNzY1NDgwNjQ@..jpg',
    subtitle: null,
    caption: null,
    id: 3442,
  },
  {
    title: 'Fish Called Wanda, A (1988)',
    description:
      'In London, four very different people team up to commit armed robbery, then try to doublecross each other for the loot.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNTQ0ODhiNjEtMDFiYi00NmZhLWJkMzQtNTBkOGQwOTliZDAxXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Comedy, Crime',
    caption: 'English, Italian, Russian, French',
    id: 9,
  },
  {
    title: 'Candidate, The (1972)',
    description:
      'Bill McKay is a candidate for the U.S. Senate from California. He has no hope of winning, so he is willing to tweak the establishment.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTYxMjY5ZWItYTRkNC00MGIxLTk0MzMtZTZhMDg0MGY2ZWU5XkEyXkFqcGdeQXVyNjE5MjUyOTM@..jpg',
    subtitle: 'Comedy, Drama',
    caption: 'English',
    id: 1236,
  },
  {
    title: 'Bonnie and Clyde (1967)',
    description: 'Bored waitress',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTViZmMwOGEtYzc4Yy00ZGQ1LWFkZDQtMDljNGZlMjAxMjhiXkEyXkFqcGdeQXVyNzM0MTUwNTY@..jpg',
    subtitle: 'Action, Biography, Crime',
    caption: 'English',
    id: 143,
  },
  {
    title: 'Old Man and the Sea, The (1958)',
    description:
      "An old Cuban fisherman's dry spell is broken when he hooks a gigantic fish that drags him out to sea. Based on Ernest Hemingway's story.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMzFiYTdmODItZGVlZi00MThlLWIzMjQtMDIwNzEyNWNhYTg1XkEyXkFqcGdeQXVyNjc1NTYyMjg@..jpg',
    subtitle: 'Adventure, Drama',
    caption: 'English',
    id: 4331,
  },
  {
    title: 'Perfect Candidate, A (1996)',
    description: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTMzMjgyMjI2N15BMl5BanBnXkFtZTcwMzg1NTUyMQ@@..jpg',
    subtitle: null,
    caption: null,
    id: 4341,
  },
  {
    title: 'Monty Python and the Holy Grail (1975)',
    description: 'King Arthur (',
    image:
      'https://m.media-amazon.com/images/M/MV5BN2IyNTE4YzUtZWU0Mi00MGIwLTgyMmQtMzQ4YzQxYWNlYWE2XkEyXkFqcGdeQXVyNjU0OTQ0OTY@..jpg',
    subtitle: 'Adventure, Comedy, Fantasy',
    caption: 'English, French, Latin',
    id: 268,
  },
  {
    title: 'Three Lives and Only One Death (Trois vies & une seule mort) (1996)',
    description:
      'Take a walk into the dreamlike world of filmmaker Raul Ruiz as he takes us to Paris for a twisting ride. Four strangely symmetrical stories unfold involving love, lust, crime, and time.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOWE5MWYwYjYtY2RlOS00Yzk1LThkYWYtZWJkNzcxMjE5NjBhXkEyXkFqcGdeQXVyMTAxMDQ0ODk@..jpg',
    subtitle: 'Comedy, Crime',
    caption: 'French',
    id: 7597,
  },
  {
    title: 'Sex, Lies, and Videotape (1989)',
    description:
      "A sexually repressed woman's husband is having an affair with her sister. The arrival of a visitor with a rather unusual fetish changes everything.",
    image:
      'https://m.media-amazon.com/images/M/MV5BNDllYWVkOTQtZjRlMC00NWFjLWI0OGEtOWY4YzU4ZjMxYzg3XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    subtitle: 'Drama',
    caption: 'English',
    id: 1242,
  },
  {
    title: "Cheech and Chong's Up in Smoke (1978)",
    description:
      'Two stoners unknowingly smuggle a van - made entirely of marijuana - from Mexico to L.A., with incompetent Sgt. Stedenko on their trail.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMmU3MTNiYjEtODQ5Yy00Y2RiLWIyMjctNGNiNmFmYTA4Y2NhXkEyXkFqcGdeQXVyNzc5MjA3OA@@..jpg',
    subtitle: 'Comedy, Music',
    caption: 'English, Spanish',
    id: 2353,
  },
  {
    title: 'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
    description:
      'In 1936, archaeologist and adventurer Indiana Jones is hired by the U.S. government to find the Ark of the Covenant before',
    image:
      'https://m.media-amazon.com/images/M/MV5BMjA0ODEzMTc1Nl5BMl5BanBnXkFtZTcwODM2MjAxNA@@..jpg',
    subtitle: 'Action, Adventure',
    caption: 'English, German, Hebrew, Spanish, Arabic, Nepali',
    id: 13,
  },
  {
    title: 'Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966)',
    description:
      'A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTQ5NDI3MTI4MF5BMl5BanBnXkFtZTgwNDQ4ODE5MDE@..jpg',
    subtitle: 'Western',
    caption: 'Italian',
    id: 536,
  },
  {
    title: 'Big Blue, The (Grand bleu, Le) (1988)',
    description:
      'The rivalry between Enzo and Jacques, two childhood friends and now world-renowned free divers, becomes a beautiful and perilous journey into oneself and the unknown.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTg5NGE1MjYtZjU3Zi00NWZhLTg4ZmEtMDI0YzMwN2Y2NDIxXkEyXkFqcGdeQXVyNTAyODkwOQ@@..jpg',
    subtitle: 'Adventure, Drama, Sport',
    caption: 'French, English, Italian',
    id: 2211,
  },
];

export const users = [
  { label: '26634', id: 0 },
  { label: '77065', id: 1 },
  { label: '87768', id: 2 },
  { label: '114069', id: 3 },
  { label: '41822', id: 4 },
  { label: '28487', id: 5 },
  { label: '43003', id: 6 },
  { label: '79702', id: 7 },
  { label: '106076', id: 8 },
  { label: '103185', id: 9 },
  { label: '33174', id: 10 },
  { label: '132954', id: 11 },
  { label: '601', id: 12 },
  { label: '8065', id: 13 },
  { label: '30589', id: 14 },
  { label: '22878', id: 15 },
  { label: '8527', id: 16 },
  { label: '96503', id: 17 },
  { label: '73227', id: 18 },
  { label: '60167', id: 19 },
  { label: '22425', id: 20 },
  { label: '110484', id: 21 },
  { label: '63770', id: 22 },
  { label: '45883', id: 23 },
  { label: '92683', id: 24 },
  { label: '4986', id: 25 },
  { label: '84523', id: 26 },
  { label: '117302', id: 27 },
  { label: '47624', id: 28 },
  { label: '134844', id: 29 },
  { label: '93002', id: 30 },
  { label: '51991', id: 31 },
  { label: '45262', id: 32 },
  { label: '134372', id: 33 },
  { label: '111507', id: 34 },
  { label: '29571', id: 35 },
  { label: '30740', id: 36 },
  { label: '118939', id: 37 },
  { label: '71423', id: 38 },
  { label: '68179', id: 39 },
  { label: '91750', id: 40 },
  { label: '3836', id: 41 },
  { label: '51151', id: 42 },
  { label: '113373', id: 43 },
  { label: '78335', id: 44 },
  { label: '33687', id: 45 },
  { label: '72686', id: 46 },
  { label: '115628', id: 47 },
  { label: '47687', id: 48 },
  { label: '75618', id: 49 },
  { label: '41205', id: 50 },
  { label: '128037', id: 51 },
  { label: '119116', id: 52 },
  { label: '125899', id: 53 },
  { label: '33432', id: 54 },
  { label: '133779', id: 55 },
  { label: '28646', id: 56 },
  { label: '124685', id: 57 },
  { label: '93537', id: 58 },
  { label: '93523', id: 59 },
  { label: '11022', id: 60 },
  { label: '40008', id: 61 },
  { label: '22018', id: 62 },
  { label: '14238', id: 63 },
  { label: '29690', id: 64 },
  { label: '113619', id: 65 },
  { label: '59868', id: 66 },
  { label: '90263', id: 67 },
  { label: '22305', id: 68 },
  { label: '33124', id: 69 },
  { label: '31772', id: 70 },
  { label: '61583', id: 71 },
  { label: '48600', id: 72 },
  { label: '111822', id: 73 },
  { label: '17874', id: 74 },
  { label: '108837', id: 75 },
  { label: '135472', id: 76 },
  { label: '24577', id: 77 },
  { label: '45597', id: 78 },
  { label: '15434', id: 79 },
  { label: '59981', id: 80 },
  { label: '76924', id: 81 },
  { label: '36685', id: 82 },
  { label: '128022', id: 83 },
  { label: '71582', id: 84 },
  { label: '48254', id: 85 },
  { label: '75998', id: 86 },
  { label: '110044', id: 87 },
  { label: '138299', id: 88 },
  { label: '38615', id: 89 },
  { label: '8127', id: 90 },
  { label: '122072', id: 91 },
  { label: '114317', id: 92 },
  { label: '34050', id: 93 },
  { label: '59194', id: 94 },
  { label: '39632', id: 95 },
  { label: '47998', id: 96 },
  { label: '33715', id: 97 },
  { label: '98431', id: 98 },
  { label: '88700', id: 99 },
  { label: '24573', id: 100 },
  { label: '97082', id: 101 },
  { label: '16830', id: 102 },
  { label: '62398', id: 103 },
  { label: '5005', id: 104 },
  { label: '50495', id: 105 },
  { label: '136348', id: 106 },
  { label: '125639', id: 107 },
  { label: '129775', id: 108 },
  { label: '4745', id: 109 },
  { label: '131303', id: 110 },
  { label: '1634', id: 111 },
  { label: '79475', id: 112 },
  { label: '2668', id: 113 },
  { label: '29123', id: 114 },
  { label: '9221', id: 115 },
  { label: '57343', id: 116 },
  { label: '106957', id: 117 },
  { label: '44571', id: 118 },
  { label: '17583', id: 119 },
  { label: '20846', id: 120 },
  { label: '121779', id: 121 },
  { label: '131955', id: 122 },
  { label: '122455', id: 123 },
  { label: '129288', id: 124 },
  { label: '13674', id: 125 },
  { label: '41260', id: 126 },
  { label: '130605', id: 127 },
  { label: '34674', id: 128 },
  { label: '26851', id: 129 },
  { label: '97186', id: 130 },
  { label: '30756', id: 131 },
  { label: '23059', id: 132 },
  { label: '108807', id: 133 },
  { label: '10183', id: 134 },
  { label: '45235', id: 135 },
  { label: '78376', id: 136 },
  { label: '11744', id: 137 },
  { label: '72539', id: 138 },
  { label: '95059', id: 139 },
  { label: '129770', id: 140 },
  { label: '90576', id: 141 },
  { label: '22115', id: 142 },
  { label: '116147', id: 143 },
  { label: '137012', id: 144 },
  { label: '21154', id: 145 },
  { label: '64144', id: 146 },
  { label: '16798', id: 147 },
  { label: '15255', id: 148 },
  { label: '25889', id: 149 },
  { label: '112767', id: 150 },
  { label: '12936', id: 151 },
  { label: '53022', id: 152 },
  { label: '123080', id: 153 },
  { label: '47098', id: 154 },
  { label: '39071', id: 155 },
  { label: '60261', id: 156 },
  { label: '41145', id: 157 },
  { label: '81141', id: 158 },
  { label: '30010', id: 159 },
  { label: '26194', id: 160 },
  { label: '128517', id: 161 },
  { label: '90509', id: 162 },
  { label: '31818', id: 163 },
  { label: '36376', id: 164 },
  { label: '91585', id: 165 },
  { label: '32174', id: 166 },
  { label: '129873', id: 167 },
  { label: '56515', id: 168 },
  { label: '46430', id: 169 },
  { label: '59586', id: 170 },
  { label: '118925', id: 171 },
  { label: '9795', id: 172 },
  { label: '61072', id: 173 },
  { label: '119794', id: 174 },
  { label: '73634', id: 175 },
  { label: '28059', id: 176 },
  { label: '29747', id: 177 },
  { label: '50344', id: 178 },
  { label: '94595', id: 179 },
  { label: '45291', id: 180 },
  { label: '74409', id: 181 },
  { label: '73423', id: 182 },
  { label: '20925', id: 183 },
  { label: '73796', id: 184 },
  { label: '119711', id: 185 },
  { label: '64285', id: 186 },
  { label: '105997', id: 187 },
  { label: '86743', id: 188 },
  { label: '130521', id: 189 },
  { label: '110396', id: 190 },
  { label: '7876', id: 191 },
  { label: '56868', id: 192 },
  { label: '42461', id: 193 },
  { label: '96120', id: 194 },
  { label: '95494', id: 195 },
  { label: '76025', id: 196 },
  { label: '5101', id: 197 },
  { label: '46160', id: 198 },
  { label: '84822', id: 199 },
  { label: '136187', id: 200 },
  { label: '76403', id: 201 },
  { label: '63289', id: 202 },
  { label: '130677', id: 203 },
  { label: '124143', id: 204 },
  { label: '124683', id: 205 },
  { label: '108551', id: 206 },
  { label: '113881', id: 207 },
  { label: '24924', id: 208 },
  { label: '40478', id: 209 },
  { label: '126080', id: 210 },
  { label: '90750', id: 211 },
  { label: '85854', id: 212 },
  { label: '210', id: 213 },
  { label: '72474', id: 214 },
  { label: '47314', id: 215 },
  { label: '100848', id: 216 },
  { label: '79978', id: 217 },
  { label: '122937', id: 218 },
  { label: '17606', id: 219 },
  { label: '46076', id: 220 },
  { label: '100362', id: 221 },
  { label: '99309', id: 222 },
  { label: '3796', id: 223 },
  { label: '35435', id: 224 },
  { label: '3918', id: 225 },
  { label: '81241', id: 226 },
  { label: '50595', id: 227 },
  { label: '132521', id: 228 },
  { label: '86751', id: 229 },
  { label: '27224', id: 230 },
  { label: '105460', id: 231 },
  { label: '71459', id: 232 },
  { label: '82637', id: 233 },
  { label: '93570', id: 234 },
  { label: '137036', id: 235 },
  { label: '3915', id: 236 },
  { label: '74259', id: 237 },
  { label: '61294', id: 238 },
  { label: '30040', id: 239 },
];

export const userSpace = [
  { label: '26634', id: 0, x: 0.77, y: 4.33, cluster: 3 },
  { label: '77065', id: 1, x: -4.49, y: 4.59, cluster: 2 },
  { label: '87768', id: 2, x: 2.62, y: 4.55, cluster: 4 },
  { label: '114069', id: 3, x: 4.3, y: 0.95, cluster: 2 },
  { label: '41822', id: 4, x: 4.97, y: 1.71, cluster: 2 },
  { label: '28487', id: 5, x: 0.01, y: 3.15, cluster: 2 },
  { label: '43003', id: 6, x: -1.95, y: 3.28, cluster: 1 },
  { label: '79702', id: 7, x: 1.98, y: -1.95, cluster: 1 },
  { label: '106076', id: 8, x: 2.57, y: 2.51, cluster: 4 },
  { label: '103185', id: 9, x: 0.42, y: 3.04, cluster: 3 },
  { label: '33174', id: 10, x: 0.09, y: 3.8, cluster: 4 },
  { label: '132954', id: 11, x: 0.51, y: 0.68, cluster: 1 },
  { label: '601', id: 12, x: -4.59, y: 1.93, cluster: 1 },
  { label: '8065', id: 13, x: 2.28, y: 4.9, cluster: 3 },
  { label: '30589', id: 14, x: 0.23, y: 1.84, cluster: 1 },
  { label: '22878', id: 15, x: -2.58, y: 1.4, cluster: 3 },
  { label: '8527', id: 16, x: 3.95, y: 0.16, cluster: 3 },
  { label: '96503', id: 17, x: 2.64, y: 3.3, cluster: 2 },
  { label: '73227', id: 18, x: -1.98, y: 1.12, cluster: 3 },
  { label: '60167', id: 19, x: 1.83, y: 1.03, cluster: 3 },
];
