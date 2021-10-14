export const models = [
  {
    key: 'knn',
    attributes: [
      {
        key: 'n',
        label: 'Neighbors',
        type: 'number',
        defaultValue: 5,
      },
    ],
  },
  {
    key: 'vasp',
    attributes: [
      {
        key: 'h',
        label: 'Some parameter',
        type: 'text',
      },
    ],
    businessRules: ['popularity', 'explore'],
  },
];

export const users = [
  {
    id: 1,
  },
  {
    id: 2,
  },
  {
    id: 3,
  },
  {
    id: 4,
  },
  {
    id: 5,
  },
  {
    id: 6,
  },
  {
    id: 7,
  },
  {
    id: 8,
  },
  {
    id: 9,
  },
  {
    id: 10,
  },
  {
    id: 11,
  },
  {
    id: 12,
  },
  {
    id: 13,
  },
  {
    id: 14,
  },
  {
    id: 15,
  },
  {
    id: 16,
  },
  {
    id: 17,
  },
  {
    id: 18,
  },
  {
    id: 19,
  },
  {
    id: 20,
  },
  {
    id: 21,
  },
  {
    id: 22,
  },
  {
    id: 23,
  },
  {
    id: 24,
  },
  {
    id: 25,
  },
  {
    id: 26,
  },
  {
    id: 27,
  },
  {
    id: 28,
  },
  {
    id: 29,
  },
  {
    id: 30,
  },
  {
    id: 31,
  },
  {
    id: 32,
  },
  {
    id: 33,
  },
  {
    id: 34,
  },
  {
    id: 35,
  },
  {
    id: 36,
  },
  {
    id: 37,
  },
  {
    id: 38,
  },
  {
    id: 39,
  },
  {
    id: 40,
  },
  {
    id: 41,
  },
  {
    id: 42,
  },
  {
    id: 43,
  },
  {
    id: 44,
  },
  {
    id: 45,
  },
  {
    id: 46,
  },
  {
    id: 47,
  },
  {
    id: 48,
  },
  {
    id: 49,
  },
  {
    id: 50,
  },
];

export const movies = [
  { id: 1, title: 'Subspecies', header: 'Horror', subtitle: 2003 },
  {
    id: 2,
    title: 'Mike Birbiglia: What I Should Have Said Was Nothing',
    header: 'Comedy',
    subtitle: 1986,
  },
  { id: 3, title: '90 Minutes (90 minutter)', header: 'Drama', subtitle: 1989 },
  { id: 4, title: 'Village, The', header: 'Drama|Mystery|Thriller', subtitle: 1991 },
  { id: 5, title: 'Everything Is Illuminated', header: 'Comedy|Drama', subtitle: 2004 },
  { id: 6, title: 'Sin of Harold Diddlebock, The', header: 'Comedy', subtitle: 2007 },
  { id: 7, title: 'Another Me', header: 'Mystery|Thriller', subtitle: 2012 },
  { id: 8, title: 'The Last Station', header: 'Drama', subtitle: 2010 },
  { id: 9, title: "I'm the One That I Want", header: 'Comedy', subtitle: 2010 },
  {
    id: 10,
    title: 'Jason Goes to Hell: The Final Friday',
    header: 'Action|Horror',
    subtitle: 2010,
  },
  { id: 11, title: 'Secretary', header: 'Comedy|Drama|Romance', subtitle: 2004 },
  { id: 12, title: 'Winter Kills', header: 'Drama', subtitle: 2010 },
  {
    id: 13,
    title: 'Tales of Ordinary Madness (Storie di Ordinaria Follia)',
    header: 'Drama',
    subtitle: 2006,
  },
  { id: 14, title: 'Violets Are Blue...', header: 'Drama|Romance', subtitle: 2010 },
  { id: 15, title: 'Best Foot Forward', header: 'Comedy|Musical', subtitle: 2003 },
  {
    id: 16,
    title: 'Resident Evil: Apocalypse',
    header: 'Action|Horror|Sci-Fi|Thriller',
    subtitle: 1996,
  },
  {
    id: 17,
    title: 'Super Inframan, The (Zhong guo chao ren)',
    header: 'Action|Fantasy|Sci-Fi',
    subtitle: 2002,
  },
  { id: 18, title: 'Bandit Queen', header: 'Drama', subtitle: 1997 },
  { id: 19, title: 'Hail Caesar', header: 'Comedy', subtitle: 2010 },
  { id: 20, title: 'Contraband', header: 'Action|Crime|Drama|Thriller', subtitle: 2006 },
  { id: 21, title: 'Gasoline (Benzina)', header: 'Crime', subtitle: 1998 },
  { id: 22, title: 'Brain, The', header: 'Horror|Sci-Fi', subtitle: 2009 },
  {
    id: 23,
    title: 'Effect of Gamma Rays on Man-in-the-Moon Marigolds, The',
    header: 'Drama',
    subtitle: 1993,
  },
  { id: 24, title: "Sovereign's Company", header: 'Drama', subtitle: 1996 },
  { id: 25, title: 'Black Gold', header: 'Documentary', subtitle: 2003 },
  { id: 26, title: 'Bullet for Joey, A', header: 'Crime|Drama|Film-Noir|Thriller', subtitle: 2008 },
  { id: 27, title: 'Little Fockers', header: 'Comedy', subtitle: 1997 },
  { id: 28, title: 'Glass House, The', header: 'Drama', subtitle: 2011 },
  { id: 29, title: 'Rita, Sue and Bob Too!', header: 'Comedy|Drama', subtitle: 2012 },
  { id: 30, title: "Love Crime (Crime d'amour)", header: 'Crime|Mystery|Thriller', subtitle: 1998 },
  {
    id: 31,
    title: 'Star Wars: Episode III - Revenge of the Sith',
    header: 'Action|Adventure|Sci-Fi',
    subtitle: 1990,
  },
  { id: 32, title: 'Lucky 7', header: 'Comedy|Romance', subtitle: 2009 },
  {
    id: 33,
    title: 'No Regrets for Our Youth (Waga seishun ni kuinashi)',
    header: 'Drama',
    subtitle: 1992,
  },
  { id: 34, title: 'Hide and Seek', header: 'Horror|Mystery|Thriller', subtitle: 1993 },
  { id: 35, title: 'Above the Rim', header: 'Crime|Drama', subtitle: 2003 },
  { id: 36, title: 'Windy Day (Tuulinen päivä)', header: 'Drama', subtitle: 2003 },
  { id: 37, title: 'Confessions of a Shopaholic', header: 'Comedy|Romance', subtitle: 2004 },
  { id: 38, title: 'Injury to One, An', header: 'Documentary', subtitle: 1996 },
  { id: 39, title: 'Incredible Burt Wonderstone, The', header: 'Comedy', subtitle: 1996 },
  { id: 40, title: 'Rent', header: 'Drama|Musical|Romance', subtitle: 1980 },
  { id: 41, title: 'Blade Runner', header: 'Action|Sci-Fi|Thriller', subtitle: 2003 },
  { id: 42, title: 'Trinity and Beyond', header: 'Documentary', subtitle: 2008 },
  { id: 43, title: "Muhammad Ali's Greatest Fight", header: 'Drama', subtitle: 2008 },
  { id: 44, title: 'Keeping Mum', header: 'Comedy|Crime', subtitle: 2004 },
  { id: 45, title: 'Eight Below', header: 'Action|Adventure|Drama|Romance', subtitle: 1994 },
  {
    id: 46,
    title: 'Rescuers, The',
    header: 'Adventure|Animation|Children|Crime|Drama',
    subtitle: 1992,
  },
  { id: 47, title: 'Story of Louis Pasteur, The', header: 'Drama', subtitle: 2011 },
  { id: 48, title: 'Sonic Outlaws', header: 'Documentary', subtitle: 2006 },
  { id: 49, title: 'Lost Souls', header: 'Drama|Horror|Thriller', subtitle: 2008 },
  {
    id: 50,
    title: 'Echoes of the Rainbow (Sui yuet san tau)',
    header: 'Comedy|Drama|Romance',
    subtitle: 2012,
  },
  { id: 51, title: 'Never Die Alone', header: 'Crime|Drama|Thriller', subtitle: 1967 },
  { id: 52, title: 'American President, The', header: 'Comedy|Drama|Romance', subtitle: 1998 },
  { id: 53, title: 'Rocket Science', header: 'Comedy|Drama', subtitle: 2007 },
  {
    id: 54,
    title: 'Captain America: The First Avenger',
    header: 'Action|Adventure|Sci-Fi|Thriller|War',
    subtitle: 2009,
  },
  { id: 55, title: 'Sitter, The', header: 'Comedy', subtitle: 1997 },
  { id: 56, title: 'DysFunktional Family', header: 'Comedy|Documentary', subtitle: 2000 },
  { id: 57, title: 'Karlsson on the Roof', header: 'Children', subtitle: 1984 },
  {
    id: 58,
    title: 'Burma Conspiracy, The (Largo Winch II)',
    header: 'Action|Adventure|Thriller',
    subtitle: 2001,
  },
  { id: 59, title: 'The Pool Boys', header: 'Comedy', subtitle: 2008 },
  { id: 60, title: 'Starbuck', header: 'Comedy', subtitle: 1986 },
  { id: 61, title: 'Me and Earl and the Dying Girl', header: 'Drama', subtitle: 1966 },
  { id: 62, title: 'American Mary', header: 'Horror|Thriller', subtitle: 2010 },
  { id: 63, title: 'Trans', header: 'Drama', subtitle: 1999 },
  { id: 64, title: 'Mág', header: 'Drama', subtitle: 2009 },
  { id: 65, title: "Pot O' Gold", header: 'Comedy|Musical', subtitle: 1998 },
  { id: 66, title: 'Gunday', header: 'Action|Crime|Drama', subtitle: 2002 },
  {
    id: 67,
    title: 'Orphanage, The (Orfanato, El)',
    header: 'Drama|Horror|Mystery|Thriller',
    subtitle: 2007,
  },
  {
    id: 68,
    title: 'Jeanne and the Perfect Guy (Jeanne et le garçon formidable)',
    header: 'Comedy|Drama|Romance',
    subtitle: 1994,
  },
  { id: 69, title: 'What! No Beer?', header: 'Comedy', subtitle: 2005 },
  { id: 70, title: 'Eye, The (Gin gwai) (Jian gui)', header: 'Thriller', subtitle: 1992 },
  { id: 71, title: "You're Telling Me!", header: 'Comedy', subtitle: 1994 },
  { id: 72, title: 'Thieves (Voleurs, Les)', header: 'Crime|Drama|Romance', subtitle: 2012 },
  { id: 73, title: 'Honeymoon in Vegas', header: 'Comedy|Romance', subtitle: 2005 },
  { id: 74, title: 'Four Stories of St. Julian ', header: 'Crime|Thriller', subtitle: 2003 },
  { id: 75, title: 'Fifty Dead Men Walking', header: 'Action|Drama|Thriller', subtitle: 1994 },
  { id: 76, title: 'Attack on the Iron Coast', header: 'Action|Drama|War', subtitle: 1987 },
  { id: 77, title: 'Airbag', header: 'Action|Comedy|Crime|Thriller', subtitle: 2011 },
  { id: 78, title: 'Future Weather', header: 'Drama', subtitle: 1994 },
  { id: 79, title: 'Man in the Saddle', header: 'Western', subtitle: 2005 },
  { id: 80, title: 'Bobby', header: 'Drama', subtitle: 1997 },
  { id: 81, title: 'Voyage to the Prehistoric Planet', header: 'Adventure|Sci-Fi', subtitle: 1996 },
  { id: 82, title: 'If You Love (Jos rakastat)', header: 'Drama|Musical|Romance', subtitle: 2003 },
  { id: 83, title: 'Lunacy (Sílení)', header: 'Animation|Horror', subtitle: 1992 },
  {
    id: 84,
    title: 'Goonies, The',
    header: 'Action|Adventure|Children|Comedy|Fantasy',
    subtitle: 1989,
  },
  { id: 85, title: 'Hide-Out', header: 'Comedy|Crime|Drama|Romance', subtitle: 1995 },
  { id: 86, title: 'Black Christmas', header: 'Action|Horror|Thriller', subtitle: 1988 },
  { id: 87, title: 'Stories We Tell', header: 'Documentary', subtitle: 1994 },
  { id: 88, title: 'Girl Who Talked to Dolphins, The', header: 'Documentary', subtitle: 1984 },
  { id: 89, title: 'Up at the Villa', header: 'Drama', subtitle: 2010 },
  { id: 90, title: 'Champion', header: 'Drama|Film-Noir|Romance', subtitle: 2007 },
  { id: 91, title: 'Zachariah', header: 'Comedy|Musical|Western', subtitle: 2009 },
  { id: 92, title: 'Complicit', header: '(no header listed)', subtitle: 1999 },
  { id: 93, title: 'Delivery Man', header: 'Comedy', subtitle: 2005 },
  {
    id: 94,
    title: 'Mortal Kombat: Annihilation',
    header: 'Action|Adventure|Fantasy',
    subtitle: 2004,
  },
  {
    id: 95,
    title: "Daffy Duck's Quackbusters",
    header: 'Animation|Children|Comedy|Horror',
    subtitle: 2006,
  },
  { id: 96, title: 'Kull the Conqueror', header: 'Action|Adventure', subtitle: 2005 },
  { id: 97, title: 'Last Elvis, The (Último Elvis, El)', header: 'Drama', subtitle: 1998 },
  { id: 98, title: 'Catch-22', header: 'Comedy|War', subtitle: 2007 },
  { id: 99, title: 'I Am (Jestem)', header: 'Drama', subtitle: 1992 },
  { id: 100, title: 'Hatchet', header: 'Comedy|Horror', subtitle: 2008 },
];
