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
  rest.get('/api/users/:splitName', (req, res, ctx) => {
    // train, validation, test
    const { splitName } = req.params;
    const interactionsFilter = req.url.searchParams.get('interactionsFilter');
    const interactionsLimit = req.url.searchParams.get('interactionsLimit');

    if (interactionsFilter && interactionsLimit) {
      const shuffledArray = shuffle(splitName === 'validation' ? vadUsers : trainUsers);
      const randIds = shuffledArray.slice(0, randomInt(3, 20));
      return res(ctx.delay(1000), ctx.json(randIds));
    }

    return res(ctx.json(splitName === 'validation' ? vadUsers : trainUsers));
  }),
  rest.post('/api/users/:splitName/describe', (req, res, ctx) => {
    const { splitName } = req.params;
    const { userIds } = req.body;
    usersDescription.topItems = shuffle(items).slice(0, 5);
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
    const fieldMask = req.url.searchParams.get('fieldMask');
    // attribute name-attribute value pairs
    const filter = req.url.searchParams.get('filter');

    if (!filter) {
      return res(ctx.status(400));
    }

    const randItems = shuffle(items).slice(0, randomInt(3, 20));
    return res(ctx.delay(1000), ctx.json(randItems));
  }),
  rest.post('/api/items/describe', (req, res, ctx) => {
    const { itemIds } = req.body;
    return res(ctx.delay(1000), ctx.json(itemsDescription));
  }),
  rest.get('/api/embeddings/:splitName/users', (req, res, ctx) => {
    // split name: train, validation, test
    const { splitName } = req.params;
    const data = splitName === 'validation' ? vadUsersEmbeddings : trainUsersEmbeddings;
    return res(ctx.delay(1500), ctx.json(data));
  }),
  rest.get('/api/embeddings/:splitName/items', (req, res, ctx) =>
    res(ctx.delay(1500), ctx.json(itemsEmbeddings))
  ),
];
