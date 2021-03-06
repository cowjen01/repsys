// eslint-disable-next-line import/no-extraneous-dependencies
import { rest } from 'msw';

import { dataset } from './data/dataset';
import { models, userMetrics, summaryMetrics, itemMetrics } from './data/models';
import { items, itemsDescription, itemsEmbeddings } from './data/items';
import {
  vadUsers,
  usersDescription,
  vadUsersEmbeddings,
  trainUsers,
  trainUsersEmbeddings,
} from './data/users';
import defaultConfig from './data/config';

function shuffle(a) {
  const b = a.slice();
  return b.sort(() => Math.random() - 0.5);
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

function randomItemsDescription() {
  const randGenres = shuffle(dataset.attributes.genres.options).slice(0, 4);
  const randCountries = shuffle(dataset.attributes.country.options).slice(0, 4);
  itemsDescription.description.genres.labels = randGenres;
  itemsDescription.description.country.labels = randCountries;
  itemsDescription.description.genres.values = Array(5)
    .fill()
    .map(() => randomInt(100, 1000));
  itemsDescription.description.country.values = Array(5)
    .fill()
    .map(() => randomInt(100, 1000));
}

export const handlers = [
  rest.get('/api/users', (req, res, ctx) => {
    // train, validation, test
    const split = req.url.searchParams.get('split');
    if (!split) return res(ctx.status(400));
    return res(ctx.json(split === 'validation' ? vadUsers : trainUsers));
  }),
  rest.get('/api/users/embeddings', (req, res, ctx) => {
    // split name: train, validation, test
    const split = req.url.searchParams.get('split');
    if (!split) return res(ctx.status(400));
    const data = split === 'validation' ? vadUsersEmbeddings : trainUsersEmbeddings;
    return res(ctx.delay(1500), ctx.json(data));
  }),
  rest.post('/api/users/search', (req, res, ctx) => {
    const { split } = req.body;
    if (!split) return res(ctx.status(400));
    const shuffledArray = shuffle(split === 'validation' ? vadUsers : trainUsers);
    const randIds = shuffledArray.slice(0, randomInt(3, 20));
    return res(ctx.delay(1000), ctx.json(randIds));
  }),
  rest.post('/api/users/describe', (req, res, ctx) => {
    usersDescription.topItems = shuffle(items).slice(0, 5);
    randomItemsDescription();
    usersDescription.itemsDescription = itemsDescription.description;
    return res(ctx.delay(1000), ctx.json(usersDescription));
  }),
  rest.get('/api/users/:userID', (req, res, ctx) => {
    const randItems = shuffle(items).slice(0, randomInt(5, 50));
    return res(
      ctx.delay(1000),
      ctx.json({
        interactions: randItems,
      })
    );
  }),
  rest.get('/api/web/config', (req, res, ctx) => res(ctx.json(defaultConfig))),
  rest.get('/api/dataset', (req, res, ctx) => res(ctx.delay(500), ctx.json(dataset))),
  rest.get('/api/models', (req, res, ctx) => res(ctx.json(models))),
  rest.get('/api/models/metrics', (req, res, ctx) =>
    // summary of the current and previous metrics
    res(ctx.delay(800), ctx.json(summaryMetrics))
  ),
  rest.get('/api/models/:modelName/metrics/user', (req, res, ctx) => {
    const { modelName } = req.params;
    const compareModel = req.url.searchParams.get('compare_againts');
    return res(ctx.delay(1000), ctx.json(userMetrics[modelName]));
  }),
  rest.get('/api/models/:modelName/metrics/item', (req, res, ctx) => {
    const { modelName } = req.params;
    return res(ctx.delay(1000), ctx.json(itemMetrics[modelName]));
  }),
  rest.post('/api/models/:modelName/predict', (req, res, ctx) => {
    const { limit } = req.body;
    const randItems = shuffle(items).slice(5, limit);
    return res(ctx.delay(1000), ctx.json(randItems));
  }),
  rest.get('/api/items', (req, res, ctx) => {
    const query = req.url.searchParams.get('query');
    if (!query) return res(ctx.status(400));
    const randItems = shuffle(items).slice(0, randomInt(3, 20));
    return res(ctx.delay(1000), ctx.json(randItems));
  }),
  rest.post('/api/items/search', (req, res, ctx) => {
    const randIds = shuffle(items)
      .slice(0, randomInt(3, 20))
      .map(({ id }) => id);
    return res(ctx.delay(1000), ctx.json(randIds));
  }),
  rest.post('/api/items/describe', (req, res, ctx) => {
    randomItemsDescription();
    return res(ctx.delay(1000), ctx.json(itemsDescription));
  }),
  rest.get('/api/items/embeddings', (req, res, ctx) => {
    const split = req.url.searchParams.get('split');
    if (!split) return res(ctx.status(400));
    return res(ctx.delay(500), ctx.json(itemsEmbeddings));
  }),
];
