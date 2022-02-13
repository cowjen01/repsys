// eslint-disable-next-line import/no-extraneous-dependencies
import { rest } from 'msw';

import { dataset } from './data/dataset';
import { models, modelsMetricsFull, modelsMetricsSummary } from './data/models';
import { items, itemsDescription, itemsEmbeddings } from './data/items';
import {
  vadUsers,
  usersDescription,
  vadUsersEmbeddings,
  trainUsers,
  trainUsersEmbeddings,
} from './data/users';

function shuffle(a) {
  const b = a.slice();
  return b.sort(() => Math.random() - 0.5);
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
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
    const { query, split } = req.body;
    const { attribute, values, range, threshold } = query;
    if (!split) return res(ctx.status(400));
    const shuffledArray = shuffle(split === 'validation' ? vadUsers : trainUsers);
    const randIds = shuffledArray.slice(0, randomInt(3, 20));
    return res(ctx.delay(1000), ctx.json(randIds));
  }),
  rest.post('/api/users/describe', (req, res, ctx) => {
    const { users: userIDs, split } = req.body;
    usersDescription.interactions.topItems = shuffle(items).slice(0, 5);
    return res(ctx.delay(1000), ctx.json(usersDescription));
  }),
  rest.get('/api/users/:userID', (req, res, ctx) => {
    const randItems = shuffle(items).slice(0, randomInt(5, 50));
    return res(
      ctx.json({
        interactions: randItems,
      })
    );
  }),
  rest.get('/api/dataset', (req, res, ctx) => res(ctx.json(dataset))),
  rest.get('/api/models', (req, res, ctx) => res(ctx.json(models))),
  rest.get('/api/models/metrics', (req, res, ctx) =>
    // summary of the current and previous metrics
    res(ctx.delay(800), ctx.json(modelsMetricsSummary))
  ),
  rest.get('/api/models/:modelName/metrics', (req, res, ctx) => {
    const { modelName } = req.params;
    return res(ctx.delay(500), ctx.json(modelsMetricsFull[modelName]));
  }),
  rest.post('/api/models/:modelName/predict', (req, res, ctx) => {
    const { modelName } = req.params;
    const { limit, params, user, interactions } = req.body;
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
    const { attribute, values, range } = req.body.query;
    const randIds = shuffle(items)
      .slice(0, randomInt(3, 20))
      .map(({ id }) => id);
    return res(ctx.delay(1000), ctx.json(randIds));
  }),
  rest.post('/api/items/describe', (req, res, ctx) => {
    const { items: itemIDs } = req.body;
    const randGenres = shuffle(dataset.attributes.genres.options).slice(0, 5);
    const randCountries = shuffle(dataset.attributes.genres.options).slice(0, 5);
    itemsDescription.attributes.genres.topValues = randGenres;
    itemsDescription.attributes.country.topValues = randCountries;
    return res(ctx.delay(1000), ctx.json(itemsDescription));
  }),
  rest.get('/api/items/embeddings', (req, res, ctx) => {
    const split = req.url.searchParams.get('split');
    if (!split) return res(ctx.status(400));
    return res(ctx.delay(500), ctx.json(itemsEmbeddings));
  }),
];
