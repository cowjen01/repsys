export const config = {
  models: [
    {
      name: 'KNN',
      params: [
        {
          name: 'n',
          label: 'Neighbors',
          type: 'number',
          default: 5,
        },
        {
          name: 'category',
          label: 'Movie category',
          type: 'select',
          options: ['comedy', 'horror'],
        },
        {
          name: 'exclude',
          label: 'Exclude history',
          type: 'bool',
          default: true,
        },
        {
          name: 'normalize',
          label: 'Normalize distances',
          type: 'bool',
          default: false,
        },
      ],
    },
    {
      name: 'VASP',
      params: [
        {
          name: 'h',
          label: 'Some parameter',
          type: 'text',
        },
      ],
    },
  ],
  dataset: {
    items: 20071,
    columns: ['title', 'about', 'image', 'genres', 'languages'],
  },
};

export const movies = [
  {
    title: 'Tom and Huck (1995)',
    about:
      'Two best friends witness a murder and embark on a series of adventures in order to prove the innocence of the man wrongly accused of the crime.',
    image:
      'https://m.media-amazon.com/images/M/MV5BN2ZkZTMxOTAtMzg1Mi00M2U0LWE2NWItZDg4YmQyZjVkMDdhXkEyXkFqcGdeQXVyNTM5NzI0NDY@..jpg',
    genres: 'Adventure, Comedy, Drama',
    languages: 'English',
    id: 6000,
  },
  {
    title: 'Dracula: Dead and Loving It (1995)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BZWQ0ZDFmYzMtZGMyMi00NmYxLWE0MGYtYzM2ZGNhMTE1NTczL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMjM5ODMxODc@..jpg',
    genres: 'Comedy, Fantasy, Horror',
    languages: 'English, German',
    id: 3380,
  },
  {
    title: 'Cutthroat Island (1995)',
    about:
      'A female pirate and her companion race against their rivals to find a hidden island that contains a fabulous treasure.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMDg2YTI0YmQtYzgwMi00Zjk4LWJkZjgtYjg0ZDE2ODUzY2RlL2ltYWdlXkEyXkFqcGdeQXVyNjQzNDI3NzY@..jpg',
    genres: 'Action, Adventure, Comedy',
    languages: 'English',
    id: 740,
  },
  {
    title: 'Sense and Sensibility (1995)',
    about:
      'Rich Mr. Dashwood dies, leaving his second wife and her three daughters poor by the rules of inheritance. The two eldest daughters are the title opposites.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNzk1MjU3MDQyMl5BMl5BanBnXkFtZTcwNjc1OTM2MQ@@..jpg',
    genres: 'Drama, Romance',
    languages: 'English, French',
    id: 391,
  },
  {
    title: 'Now and Then (1995)',
    about: 'Four 12-year-old girls grow up together during an eventful small-town summer in 1970.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM2MDQ1YjUtMGM0NC00NmFlLTljMDktZjJiNWRhMWYxOWYyXkEyXkFqcGdeQXVyNjgzMjI4ODE@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'English',
    id: 2874,
  },
  {
    title: "Mr. Holland's Opus (1995)",
    about: 'A frustrated composer finds fulfillment as a high school music teacher.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDZhNDRlZjAtYzdhNy00ZjU1LWFlMDYtNjA5NjliM2Y5ZmVjL2ltYWdlXkEyXkFqcGdeQXVyNjE5MjUyOTM@..jpg',
    genres: 'Drama, Music',
    languages: 'English, American Sign Language',
    id: 89,
  },
  {
    title: 'Kicking and Screaming (1995)',
    about:
      'A bunch of guys hang around their college for months after graduation, continuing a life much like the one before graduation.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNWU2YjdlN2ItNTk2OS00MzMwLTlhYjctNDI0MDI0NTQ3OWY0XkEyXkFqcGdeQXVyNzI1NzMxNzM@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'English',
    id: 5235,
  },
  {
    title: 'Angels and Insects (1995)',
    about: 'In the 1800s a naturalist marries into a family of British country gentry.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZTc1MzY1ODAtMDhlMS00NjgyLTlkNTEtZTUwYTM4MzFkNWNmXkEyXkFqcGdeQXVyNDE5MTU2MDE@..jpg',
    genres: 'Drama, Romance',
    languages: 'English',
    id: 1025,
  },
  {
    title: 'Muppet Treasure Island (1996)',
    about: "The Muppets' twist on the classic tale.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMTlmNzhiMWEtOWVjZC00NmM0LTgxNDItMDJmYTkxYTZkY2FjXkEyXkFqcGdeQXVyNTUyMzE4Mzg@..jpg',
    genres: 'Action, Adventure, Comedy',
    languages: 'English',
    id: 907,
  },
  {
    title: 'Before and After (1996)',
    about:
      'Two parents deal with the effects when their son is accused of murdering his girlfriend.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOWJmODIwYWUtNTNkYy00Njk4LWJkODktMzA5ZTdkMTZhZmZiXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    genres: 'Crime, Drama, Mystery',
    languages: 'English',
    id: 3648,
  },
  {
    title: "Young Poisoner's Handbook, The (1995)",
    about:
      'This film is based on a true story about a British teenager who allegedly poisoned family, friends, and co-workers. Graham is highly intelligent, but completely amoral. He becomes ...',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTg0MTE5OTMzNV5BMl5BanBnXkFtZTcwMDM0MzkzMQ@@..jpg',
    genres: 'Crime, Drama',
    languages: 'English',
    id: 4028,
  },
  {
    title: 'Up Close and Personal (1996)',
    about:
      'An ambitious young woman, determined to build a career in television journalism, gets good advice from her first boss, and they fall in love.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZTZjYmNkZTYtOTA0Zi00NDcyLWI2YzQtOTgyZjQ2YzM5Y2E2XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Drama, Romance',
    languages: 'English',
    id: 1988,
  },
  {
    title: 'Amazing Panda Adventure, The (1995)',
    about:
      'A young American boy visiting in China helps his zoologist father rescue a panda cub from unscrupulous poachers while his panda reserve is threatened with closure from officious bureaucrats.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZmUyZWJmZDktMjE0Yy00MjUwLTlmMmUtNTk0YjFkODUxZTVkXkEyXkFqcGdeQXVyNzc5MjA3OA@@..jpg',
    genres: 'Adventure, Drama, Family',
    languages: 'English',
    id: 2814,
  },
  {
    title: 'Moonlight and Valentino (1995)',
    about:
      'A young widow still grieving over the death of her husband finds herself being comforted by a local housepainter.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYjBhZDFhYWMtYTA0MS00NTZkLWI2YTAtMGJmZmVlNmE3YmE1XkEyXkFqcGdeQXVyMTA0MjU0Ng@@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'English',
    id: 7195,
  },
  {
    title: 'Death and the Maiden (1994)',
    about:
      'A political activist is convinced that her guest is a man who once tortured her for the government.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTgzZTcwNDItYjhhMy00NmViLWFjZjEtZGQ5YWQxYzk0OWU4XkEyXkFqcGdeQXVyMjgxMzgyNjI@..jpg',
    genres: 'Drama, Mystery, Thriller',
    languages: 'English',
    id: 1148,
  },
  {
    title: 'Dumb & Dumber (Dumb and Dumber) (1994)',
    about:
      'After a woman leaves a briefcase at the airport terminal, a dumb limo driver and his dumber friend set out on a hilarious cross-country road trip to Aspen to return it.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDQwMjNiMTQtY2UwYy00NjhiLTk0ZWEtZWM5ZWMzNGFjNTVkXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Comedy',
    languages: 'English, Swedish, German',
    id: 715,
  },
  {
    title: 'Pushing Hands (Tui shou) (1992)',
    about: 'All the while, Master Chu tries to find his place in the foreign American world.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMmM1ODExNjctMTg4YS00NTg1LWI5OGEtZjExMjFlMWZmN2VhXkEyXkFqcGdeQXVyNjU1MDM2NjY@..jpg',
    genres: 'Comedy, Drama',
    languages: 'Mandarin, English',
    id: 7576,
  },
  {
    title: 'Quick and the Dead, The (1995)',
    about:
      "A female gunfighter returns to a frontier town where a dueling tournament is being held, which she enters in an effort to avenge her father's death.",
    image:
      'https://m.media-amazon.com/images/M/MV5BOTI2ZTZmMmItMmM3YS00ZjUwLWJiODMtMmRjMWM4NDE0OWFhXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Action, Romance, Thriller',
    languages: 'English, Spanish',
    id: 1228,
  },
  {
    title: 'Strawberry and Chocolate (Fresa y chocolate) (1993)',
    about:
      'This Oscar nominated film is the story of two men who are opposites, one gay, the other straight, one a fierce communist, the other a fierce individualist, one suspicious, the other accepting, and how they come to love each other.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMzhjMDQ1YTctNTNmNS00NDZjLWFjYjUtM2FhZTA2NmYxZjJlXkEyXkFqcGdeQXVyMTA0MjU0Ng@@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'Spanish, English, French',
    id: 1927,
  },
  {
    title: 'Clear and Present Danger (1994)',
    about:
      'CIA Analyst Jack Ryan is drawn into an illegal war fought by the US government against a Colombian drug cartel.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNDczOWNiMmEtZjA4MS00NDMzLWExNTktYjc0MGU0YTQ3ZDExXkEyXkFqcGdeQXVyNjU0OTQ0OTY@..jpg',
    genres: 'Action, Crime, Drama',
    languages: 'English',
    id: 356,
  },
  {
    title: 'Four Weddings and a Funeral (1994)',
    about:
      'Over the course of five social occasions, a committed bachelor must consider the notion that he may have discovered love.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTMyNzg2NzgxNV5BMl5BanBnXkFtZTcwMTcxNzczNA@@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'English, British Sign Language',
    id: 357,
  },
  {
    title: 'Mrs. Parker and the Vicious Circle (1994)',
    about:
      'Dorothy Parker remembers the heyday of the Algonquin Round Table, a circle of friends whose barbed wit, like hers, was fueled by alcohol and flirted with despair.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNWUwMDdhZWMtOGZkYy00ZTg2LTgwNTktZDVhZDM1MmFhODI4XkEyXkFqcGdeQXVyMjA0MzYwMDY@..jpg',
    genres: 'Biography, Drama',
    languages: 'English',
    id: 1094,
  },
  {
    title: 'Frank and Ollie (1995)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMzc5M2NkNTYtNDg3NS00NTQ1LWJjMzYtZjhmMGI1NzkzNGY0XkEyXkFqcGdeQXVyMTQ3Njg3MQ@@..jpg',
    genres: null,
    languages: null,
    id: 10914,
  },
  {
    title: 'Highlander III: The Sorcerer (a.k.a. Highlander: The Final Dimension) (1994)',
    about:
      'Deceived that he had won the Prize, Connor MacLeod awakens from a peaceful life when an entombed immortal magician comes seeking the Highlander.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNzZmNGNmMzgtMjA2YS00YzY3LThmMjYtOTVmMzg4ZGY2YWJjXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Action, Fantasy, Romance',
    languages: 'English',
    id: 2304,
  },
  {
    title: 'Cops and Robbersons (1994)',
    about:
      'A counterfeiter with a habit of "eliminating" the competition moves in next door to the Robbersons. Two cops move in with the Robbersons for a stakeout.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMjcxM2VkZDEtYmExZi00ODRhLWI3NGItNjZiM2IxOGQxODM5XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Comedy, Crime, Thriller',
    languages: 'English',
    id: 2309,
  },
  {
    title: 'Dazed and Confused (1993)',
    about:
      'The adventures of high school and junior high students on the last day of school in May 1976.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM5MDY5MDQyOV5BMl5BanBnXkFtZTgwMzM3NzMxMDE@..jpg',
    genres: 'Comedy',
    languages: 'English',
    id: 1016,
  },
  {
    title: 'Flesh and Bone (1993)',
    about:
      "Decades later, a son of a killer falls in love with a girl, whose family's horrifying murder he saw in childhood.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMzEzNTA3N2MtNzEyNC00ZDA3LTkwNGMtYWEyODRlNjZhZDQ0XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Drama, Mystery, Romance',
    languages: 'English',
    id: 6245,
  },
  {
    title: 'Orlando (1992)',
    about:
      'After Queen Elizabeth I commands him not to grow old, a young nobleman struggles with love and his place in the world.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYmY1OTA3MjAtYjQxOC00OTlkLWExZWQtMjc3ZjExOWFhM2UwXkEyXkFqcGdeQXVyMTA0MjU0Ng@@..jpg',
    genres: 'Biography, Drama, Fantasy',
    languages: 'English, French',
    id: 1972,
  },
  {
    title: 'Radioland Murders (1994)',
    about: 'A series of mysterious crimes confuses existence of a radio network.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMWRkYzc3YjYtNWY3Yy00Y2FmLTgyOTYtM2U2ZDg0MDA0MTIwXkEyXkFqcGdeQXVyNDk3NzU2MTQ@..jpg',
    genres: 'Comedy, Crime, Drama',
    languages: 'English',
    id: 6727,
  },
  {
    title: 'Shadowlands (1993)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTE2MGEzMDctZTZlMi00MjY1LWI5NmQtYmJlZGJiYjkwNWQ5XkEyXkFqcGdeQXVyMTMxMTY0OTQ@..jpg',
    genres: 'Biography, Drama, Romance',
    languages: 'English',
    id: 397,
  },
  {
    title: 'Tough and Deadly (1995)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BZGU5ZDkxZWEtOWI0My00NGU4LTk4Y2EtOTFkNDcwMTNhZmM2XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: null,
    languages: null,
    id: 11166,
  },
  {
    title: 'Snow White and the Seven Dwarfs (1937)',
    about:
      'Exiled into the dangerous forest by her wicked stepmother, a princess is rescued by seven dwarf miners who make her part of their household.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTQwMzE2Mzc4M15BMl5BanBnXkFtZTcwMTE4NTc1Nw@@..jpg',
    genres: 'Animation, Family, Fantasy',
    languages: 'English',
    id: 257,
  },
  {
    title: 'Beauty and the Beast (1991)',
    about:
      "A prince cursed to spend his days as a hideous monster sets out to regain his humanity by earning a young woman's love.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMzE5MDM1NDktY2I0OC00YWI5LTk2NzUtYjczNDczOWQxYjM0XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Animation, Family, Fantasy',
    languages: 'English, French',
    id: 258,
  },
  {
    title: 'Love and a .45 (1994)',
    about:
      'A small time crook flees to Mexico to evade the authorities, loan sharks, and his murderous ex-partner with only his fiancé and a trusted Colt .45.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYTQ1YzE3NzUtOTZmNC00NzE0LWJiZmUtMjhkMjU1MmVhNTZmXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    genres: 'Crime, Romance, Thriller',
    languages: 'English',
    id: 8893,
  },
  {
    title: 'Candyman: Farewell to the Flesh (1995)',
    about:
      'The Candyman arrives in New Orleans and sets his sights on a young woman whose family was ruined by the immortal killer years before.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDA2ZWE2YTctN2JiNi00NjdmLThhYWQtN2JjY2M4MTNhM2I5XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Horror, Thriller',
    languages: 'English',
    id: 7580,
  },
  {
    title: 'Bread and Chocolate (Pane e cioccolata) (1973)',
    about:
      'Italian immigrant Nino steadfastly tries to become a member of Swiss Society no matter how awful his situation becomes.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMWFjYjNiZjItMjU3Yy00YTk3LWFhZmMtN2NlZTZjYTE1YTIzXkEyXkFqcGdeQXVyNjg3MTIwODI@..jpg',
    genres: 'Comedy, Drama',
    languages: 'Italian, German, English, French, Greek, Spanish',
    id: 4183,
  },
  {
    title: 'Thin Line Between Love and Hate, A (1996)',
    about:
      "An observable, fast-talking party man Darnell Wright, gets his punishment when one of his conquests takes it personally and comes back for revenge in this 'Fatal Attraction'-esque comic ...",
    image:
      'https://m.media-amazon.com/images/M/MV5BZTAwOWRlYzMtZjM0Yy00NTU3LWFkMzAtOTVlMDQ4NDVkZTUxXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    genres: 'Comedy, Crime, Drama',
    languages: 'English',
    id: 7581,
  },
  {
    title: 'Land and Freedom (Tierra y libertad) (1995)',
    about:
      'David is an unemployed communist that comes to Spain in 1937 during the civil war to enroll the republicans and defend the democracy against the fascists. He makes friends between the soldiers.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMmE2ZmE0ZDUtYzljOS00ZWZmLWIyNmYtNzkwNGM4MGNlMWY3L2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTYxNjkxOQ@@..jpg',
    genres: 'Drama, War',
    languages: 'English, Spanish, Catalan',
    id: 1973,
  },
  {
    title: 'Jack and Sarah (1995)',
    about: 'A young American woman becomes a nanny in the home of a recent British widower.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNTI0YzJjNWQtZjAwZi00OTNmLThiNDgtODY2M2QxZGI4NTAwXkEyXkFqcGdeQXVyMTczNjQwOTY@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'English',
    id: 2322,
  },
  {
    title: 'Moll Flanders (1996)',
    about:
      'The daughter of a thief, young Moll is placed in the care of a nunnery after the execution of her mother. However, the actions of an abusive Priest lead Moll to rebel as a teenager, ...',
    image:
      'https://m.media-amazon.com/images/M/MV5BZDZlODcwNGUtN2JhNi00Njc2LWJmZjYtNWI2ZWE1MTUyNDFjXkEyXkFqcGdeQXVyNjMwMjk0MTQ@..jpg',
    genres: 'Drama, Romance',
    languages: 'English',
    id: 1941,
  },
  {
    title: 'James and the Giant Peach (1996)',
    about:
      'An orphan who lives with his two cruel aunts befriends anthropomorphic bugs who live inside a giant peach, and they embark on a journey to New York City.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTNkNWIwNGUtNTJlOC00NDU3LTk0NWEtNjNjNDM4NzRiNThkXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Animation, Adventure, Family',
    languages: 'English',
    id: 1382,
  },
  {
    title: 'Kids in the Hall: Brain Candy (1996)',
    about:
      "A pharmaceutical scientist creates a pill that makes people remember their happiest memory, and although it's successful, it has unfortunate side effects.",
    image:
      'https://m.media-amazon.com/images/M/MV5BZDJlZDE4N2UtMjhjMC00ZWYyLWFiNjItZjI1NjY0MDE3YjE4XkEyXkFqcGdeQXVyNzc5MjA3OA@@..jpg',
    genres: 'Comedy',
    languages: 'English',
    id: 3300,
  },
  {
    title: 'Mulholland Falls (1996)',
    about:
      "In 1950's Los Angeles, a special crime squad of the LAPD investigates the murder of a young woman.",
    image:
      'https://m.media-amazon.com/images/M/MV5BOTcwMDE3MTQ5OV5BMl5BanBnXkFtZTcwMzA0MDQ1NA@@..jpg',
    genres: 'Crime, Drama, Mystery',
    languages: 'English',
    id: 1942,
  },
  {
    title: 'Of Love and Shadows (1994)',
    about:
      "Chile 1973 is ruled by the dictator Pinochet. The wealthy don't see the violence, terror, executions etc. including Irene. She's engaged to an officer in the fascist military. She meets Francisco who opens her eyes to truth and love.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMTIwOTMxMjk4OV5BMl5BanBnXkFtZTcwMzUzODkyMQ@@..jpg',
    genres: 'Drama, Romance, Thriller',
    languages: 'English',
    id: 7289,
  },
  {
    title: 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
    about:
      'An insane general triggers a path to nuclear holocaust that a War Room full of politicians and generals frantically tries to stop.',
    image:
      'https://m.media-amazon.com/images/M/MV5BZWI3ZTMxNjctMjdlNS00NmUwLWFiM2YtZDUyY2I3N2MxYTE0XkEyXkFqcGdeQXVyNzkwMjQ5NzM@..jpg',
    genres: 'Comedy',
    languages: 'English, Russian',
    id: 276,
  },
  {
    title: 'Carmen Miranda: Bananas Is My Business (1994)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM4ODQyMzM5MV5BMl5BanBnXkFtZTcwNDYzMjYxMQ@@..jpg',
    genres: null,
    languages: null,
    id: 10816,
  },
  {
    title: 'Marlene Dietrich: Shadow and Light (1996)',
    about: null,
    image: 'Error',
    genres: null,
    languages: null,
    id: 9842,
  },
  {
    title: 'Last Klezmer: Leopold Kozlowski, His Life and Music, The (1994)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTk2NjAwNTM1Ml5BMl5BanBnXkFtZTcwNDEwNDkyMQ@@..jpg',
    genres: null,
    languages: null,
    id: 7586,
  },
  {
    title: "My Life and Times With Antonin Artaud (En compagnie d'Antonin Artaud) (1993)",
    about:
      'May, 1946, in Paris young poet Jacques Prevel meets Antonin Artaud, the actor, artist and writer just released from a mental asylum. Over ten months, we follow the mad Artaud from his cruel...',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTM2MjE5MTQ5MF5BMl5BanBnXkFtZTcwOTMzNzcyMQ@@..jpg',
    genres: 'Biography, Drama',
    languages: 'French',
    id: 5879,
  },
  {
    title: 'Walking and Talking (1996)',
    about:
      "Just as Amelia thinks she's over her anxiety and insecurity, her best friend announces her engagement, bringing her anxiety and insecurity right back.",
    image:
      'https://m.media-amazon.com/images/M/MV5BODVjODIyNDYtZTU2Ny00MDIxLTg1ZTEtYjJlNDkyNDEzNDcwXkEyXkFqcGdeQXVyMTMxMTY0OTQ@..jpg',
    genres: 'Comedy, Drama, Romance',
    languages: 'English',
    id: 5151,
  },
  {
    title: 'Lotto Land (1995)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTkwMDUxMTQyNl5BMl5BanBnXkFtZTgwMzIwNDgwMzE@..jpg',
    genres: null,
    languages: null,
    id: 10167,
  },
  {
    title: 'Crows and Sparrows (Wuya yu maque) (1949)',
    about:
      'A story of a corrupt party official who attempts to sell an apartment building he has appropriated from the original owner and the struggles of the tenants to prevent themselves being thrown onto the street.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMWUyYTk0MjItZmZmZS00NmI3LTk5NzUtYjJkYTBiNzNmMDZmXkEyXkFqcGdeQXVyNzI1NzMxNzM@..jpg',
    genres: 'Drama',
    languages: 'Mandarin',
    id: 10171,
  },
  {
    title: 'Island of Dr. Moreau, The (1996)',
    about:
      'After being rescued and brought to an island, a man discovers that its inhabitants are experimental animals being turned into strange-looking humans, all of it the work of a visionary doctor.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYjQxNmM5ODItMGM3Yi00N2VlLTkwNTEtNDZkNzUyYzA3MmIxXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Horror, Sci-Fi, Thriller',
    languages: 'English, Indonesian',
    id: 1904,
  },
  {
    title: 'Land Before Time III: The Time of the Great Giving (1995)',
    about: null,
    image: 'https://m.media-amazon.com/images/M/MV5BMTI1OTU3NjkwMl5BMl5BanBnXkFtZTYwOTU2MDg5..jpg',
    genres: null,
    languages: null,
    id: 5223,
  },
  {
    title: 'Band Wagon, The (1953)',
    about:
      'A pretentiously artistic director is hired for a new Broadway musical and changes it beyond recognition.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNGUxZmJkZTgtMmI1Ny00Mzg3LWFlY2QtMjNkM2QwZGVhYTQyL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyNjc1NTYyMjg@..jpg',
    genres: 'Comedy, Musical, Romance',
    languages: 'English, French, German',
    id: 3557,
  },
  {
    title: 'Ghost and Mrs. Muir, The (1947)',
    about:
      'In 1900, a young widow finds her seaside cottage is haunted and forms a unique relationship with the ghost.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNzYxMjIyMmYtMDI2OS00YTgyLWExZWMtODdkZTk1NzRlODgzXkEyXkFqcGdeQXVyMTAwMzUyOTc@..jpg',
    genres: 'Comedy, Drama, Fantasy',
    languages: 'English',
    id: 2338,
  },
  {
    title: 'Angel and the Badman (1947)',
    about:
      'Quirt Evans, an all round bad guy, is nursed back to health and sought after by Penelope Worth, a Quaker girl. He eventually finds himself having to choose between his world and the world Penelope lives in.',
    image:
      'https://m.media-amazon.com/images/M/MV5BYWI1NzdhYTMtZGE2ZS00ZjJjLThmZWQtNjY4ODcxNWM2YjQ3XkEyXkFqcGdeQXVyMTI1NDQ4NQ@@..jpg',
    genres: 'Romance, Western',
    languages: 'English',
    id: 5420,
  },
  {
    title: 'Last Man Standing (1996)',
    about:
      'A drifting gunslinger-for-hire finds himself in the middle of an ongoing war between the Irish and Italian mafia in a Prohibition era ghost town.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOThkYmJjYTMtOWMzNC00ZjQ4LWI4NzAtYjRlMDA3ZWMyYWRmXkEyXkFqcGdeQXVyNDIyMjczNjI@..jpg',
    genres: 'Action, Crime, Drama',
    languages: 'English, Spanish',
    id: 2798,
  },
  {
    title: 'Winnie the Pooh and the Blustery Day (1968)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BODgyNjM3NWUtZDAxMi00NzRhLWE0NmMtMjIxYTY1MWJlZTQyXkEyXkFqcGdeQXVyNzY1NDgwNjQ@..jpg',
    genres: null,
    languages: null,
    id: 406,
  },
  {
    title: 'Bedknobs and Broomsticks (1971)',
    about:
      'An apprentice witch, three kids and a cynical magician conman search for the missing component to a magic spell to be used in the defense of Britain in World War II.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTUxMTY3MTE5OF5BMl5BanBnXkFtZTgwNTQ0ODgxMzE@..jpg',
    genres: 'Animation, Adventure, Comedy',
    languages: 'English, German',
    id: 4781,
  },
  {
    title: 'Alice in Wonderland (1951)',
    about:
      'Alice stumbles into the world of Wonderland. Will she get home? Not if the Queen of Hearts has her way.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMTgyMjM2NTAxMF5BMl5BanBnXkFtZTgwNjU1NDc2MTE@..jpg',
    genres: 'Animation, Adventure, Family',
    languages: 'English',
    id: 1917,
  },
  {
    title: 'Fox and the Hound, The (1981)',
    about:
      'A little fox named Tod, and Copper, a hound puppy, vow to be best buddies forever. But as Copper grows into a hunting dog, their unlikely friendship faces the ultimate test.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMDQzZWRlYjctMjE0ZS00NWU5LWE3NzktYzZiNzJiM2UxMDczXkEyXkFqcGdeQXVyNjUwNzk3NDc@..jpg',
    genres: 'Animation, Adventure, Drama',
    languages: 'English',
    id: 1907,
  },
  {
    title: 'Ghost and the Darkness, The (1996)',
    about:
      'A bridge engineer and an experienced old hunter begin a hunt for two lions after they start attacking local construction workers.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNWQ4NDRiMWItNGI5Yi00N2U1LTlkMGQtM2VjMzdkZTU0YzYyXkEyXkFqcGdeQXVyNTc1NTQxODI@..jpg',
    genres: 'Adventure, Drama, Thriller',
    languages: 'English, Hindi',
    id: 2152,
  },
  {
    title: 'Aladdin and the King of Thieves (1996)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BODFkMjE5YzAtMDFkOC00ZDNhLTkwNmQtODk1Y2VhNThlNWJhXkEyXkFqcGdeQXVyNzY1NDgwNjQ@..jpg',
    genres: null,
    languages: null,
    id: 3442,
  },
  {
    title: 'Fish Called Wanda, A (1988)',
    about:
      'In London, four very different people team up to commit armed robbery, then try to doublecross each other for the loot.',
    image:
      'https://m.media-amazon.com/images/M/MV5BNTQ0ODhiNjEtMDFiYi00NmZhLWJkMzQtNTBkOGQwOTliZDAxXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Comedy, Crime',
    languages: 'English, Italian, Russian, French',
    id: 9,
  },
  {
    title: 'Candidate, The (1972)',
    about:
      'Bill McKay is a candidate for the U.S. Senate from California. He has no hope of winning, so he is willing to tweak the establishment.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTYxMjY5ZWItYTRkNC00MGIxLTk0MzMtZTZhMDg0MGY2ZWU5XkEyXkFqcGdeQXVyNjE5MjUyOTM@..jpg',
    genres: 'Comedy, Drama',
    languages: 'English',
    id: 1236,
  },
  {
    title: 'Bonnie and Clyde (1967)',
    about: 'Bored waitress',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTViZmMwOGEtYzc4Yy00ZGQ1LWFkZDQtMDljNGZlMjAxMjhiXkEyXkFqcGdeQXVyNzM0MTUwNTY@..jpg',
    genres: 'Action, Biography, Crime',
    languages: 'English',
    id: 143,
  },
  {
    title: 'Old Man and the Sea, The (1958)',
    about:
      "An old Cuban fisherman's dry spell is broken when he hooks a gigantic fish that drags him out to sea. Based on Ernest Hemingway's story.",
    image:
      'https://m.media-amazon.com/images/M/MV5BMzFiYTdmODItZGVlZi00MThlLWIzMjQtMDIwNzEyNWNhYTg1XkEyXkFqcGdeQXVyNjc1NTYyMjg@..jpg',
    genres: 'Adventure, Drama',
    languages: 'English',
    id: 4331,
  },
  {
    title: 'Perfect Candidate, A (1996)',
    about: null,
    image:
      'https://m.media-amazon.com/images/M/MV5BMTMzMjgyMjI2N15BMl5BanBnXkFtZTcwMzg1NTUyMQ@@..jpg',
    genres: null,
    languages: null,
    id: 4341,
  },
  {
    title: 'Monty Python and the Holy Grail (1975)',
    about: 'King Arthur (',
    image:
      'https://m.media-amazon.com/images/M/MV5BN2IyNTE4YzUtZWU0Mi00MGIwLTgyMmQtMzQ4YzQxYWNlYWE2XkEyXkFqcGdeQXVyNjU0OTQ0OTY@..jpg',
    genres: 'Adventure, Comedy, Fantasy',
    languages: 'English, French, Latin',
    id: 268,
  },
  {
    title: 'Three Lives and Only One Death (Trois vies & une seule mort) (1996)',
    about:
      'Take a walk into the dreamlike world of filmmaker Raul Ruiz as he takes us to Paris for a twisting ride. Four strangely symmetrical stories unfold involving love, lust, crime, and time.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOWE5MWYwYjYtY2RlOS00Yzk1LThkYWYtZWJkNzcxMjE5NjBhXkEyXkFqcGdeQXVyMTAxMDQ0ODk@..jpg',
    genres: 'Comedy, Crime',
    languages: 'French',
    id: 7597,
  },
  {
    title: 'Sex, Lies, and Videotape (1989)',
    about:
      "A sexually repressed woman's husband is having an affair with her sister. The arrival of a visitor with a rather unusual fetish changes everything.",
    image:
      'https://m.media-amazon.com/images/M/MV5BNDllYWVkOTQtZjRlMC00NWFjLWI0OGEtOWY4YzU4ZjMxYzg3XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg',
    genres: 'Drama',
    languages: 'English',
    id: 1242,
  },
  {
    title: "Cheech and Chong's Up in Smoke (1978)",
    about:
      'Two stoners unknowingly smuggle a van - made entirely of marijuana - from Mexico to L.A., with incompetent Sgt. Stedenko on their trail.',
    image:
      'https://m.media-amazon.com/images/M/MV5BMmU3MTNiYjEtODQ5Yy00Y2RiLWIyMjctNGNiNmFmYTA4Y2NhXkEyXkFqcGdeQXVyNzc5MjA3OA@@..jpg',
    genres: 'Comedy, Music',
    languages: 'English, Spanish',
    id: 2353,
  },
  {
    title: 'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
    about:
      'In 1936, archaeologist and adventurer Indiana Jones is hired by the U.S. government to find the Ark of the Covenant before',
    image:
      'https://m.media-amazon.com/images/M/MV5BMjA0ODEzMTc1Nl5BMl5BanBnXkFtZTcwODM2MjAxNA@@..jpg',
    genres: 'Action, Adventure',
    languages: 'English, German, Hebrew, Spanish, Arabic, Nepali',
    id: 13,
  },
  {
    title: 'Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966)',
    about:
      'A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTQ5NDI3MTI4MF5BMl5BanBnXkFtZTgwNDQ4ODE5MDE@..jpg',
    genres: 'Western',
    languages: 'Italian',
    id: 536,
  },
  {
    title: 'Big Blue, The (Grand bleu, Le) (1988)',
    about:
      'The rivalry between Enzo and Jacques, two childhood friends and now world-renowned free divers, becomes a beautiful and perilous journey into oneself and the unknown.',
    image:
      'https://m.media-amazon.com/images/M/MV5BOTg5NGE1MjYtZjU3Zi00NWZhLTg4ZmEtMDI0YzMwN2Y2NDIxXkEyXkFqcGdeQXVyNTAyODkwOQ@@..jpg',
    genres: 'Adventure, Drama, Sport',
    languages: 'French, English, Italian',
    id: 2211,
  },
];

export const users = [
  105696, 11200, 112758, 51254, 128532, 111006, 131124, 1617, 89659, 62157, 29614, 67889, 67928,
  99266, 30492, 111299, 133935, 116347, 126563, 53074, 114696, 11268, 132150, 69084, 8535, 39813,
  66900, 129764, 36222, 34310, 26773, 41321, 53183, 30999, 48513, 24236, 77487, 6698, 92946, 60078,
  100727, 56230, 83959, 7506, 11816, 15880, 34815, 117329, 86297, 82987, 67071, 81366, 120144,
  54520, 31151, 96046, 123968, 103447, 23133, 91439, 12494, 36267, 26901, 98242, 21607, 32832,
  45261, 65010,
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
