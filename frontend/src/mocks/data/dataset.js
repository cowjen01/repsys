export const dataset = {
  attributes: {
    title: {
      dtype: 'title',
    },
    about: {
      dtype: 'string',
    },
    image: {
      dtype: 'string',
    },
    year: {
      dtype: 'number',
      bins: [0, 1990, 2000, 2010, 2015, 2020],
    },
    genres: {
      dtype: 'tags',
      options: ['Adventure', 'Animation', 'Children', 'Comedy', 'Musical', 'Crime', 'Thriller'],
    },
    languages: {
      dtype: 'tags',
      options: [
        'Nepali',
        'Haitian Creole',
        'Gujarati',
        'Māori',
        'Montenegrin',
        'Lao',
        'Moldovan',
        'Swati',
        'Somali',
        'Punjabi',
      ],
    },
    country: {
      dtype: 'category',
      options: [
        'China',
        'United States',
        'Indonesia',
        'Russia',
        'Poland',
        'Indonesia',
        'Japan',
        'China',
        'Indonesia',
        'Russia',
      ],
    },
  },
};
