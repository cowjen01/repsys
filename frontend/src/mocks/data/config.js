export default {
  recommenders: [
    {
      name: 'New Recommender',
      itemsPerPage: 4,
      itemsLimit: 20,
      model: 'knn',
      modelParams: { neighbors: 11, category: 'Crime', normalize: true },
    },
    {
      name: 'New Recommender - copy',
      itemsPerPage: 4,
      itemsLimit: 20,
      model: 'knn',
      modelParams: { neighbors: 5, category: '', normalize: false },
    },
  ],
  mappings: {
    title: 'title',
    subtitle: 'year',
    caption: 'country',
    image: 'image',
    content: 'about',
  },
};
