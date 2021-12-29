// eslint-disable-next-line import/no-extraneous-dependencies
import { rest } from 'msw';

import { config, movies, users } from './data';

function shuffle(array) {
  return array.sort(() => Math.random() - 0.5);
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

export const handlers = [
  // rest.get('/api/metrics', (req, res, ctx) => {
  //   const model = req.url.searchParams.get('model');
  //   return res(ctx.status(200), ctx.json(metrics[model]));
  // }),
  rest.get('/api/config', (req, res, ctx) => res(ctx.status(200), ctx.json(config))),
  rest.get('/api/items', (req, res, ctx) => res(ctx.status(200), ctx.json(movies))),
  rest.get('/api/users', (req, res, ctx) => res(ctx.status(200), ctx.json(users))),
  rest.post('/api/predict', (req, res, ctx) => {
    const { limit } = req.body;
    shuffle(movies);
    return res(ctx.status(200), ctx.json(movies.slice(0, limit)));
  }),
  rest.get('/api/interactions', (req, res, ctx) => {
    shuffle(movies);
    return res(ctx.status(200), ctx.json(movies.slice(0, randomInt(5, 50))));
  }),
];
