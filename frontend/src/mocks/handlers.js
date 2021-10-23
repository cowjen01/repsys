// eslint-disable-next-line import/no-extraneous-dependencies
import { rest } from 'msw';

import { models, movies, users } from './data';

function shuffle(array) {
  return array.sort(() => Math.random() - 0.5);
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

export const handlers = [
  rest.get('/api/models', (req, res, ctx) => res(ctx.status(200), ctx.json(models))),
  rest.get('/api/users', (req, res, ctx) => res(ctx.status(200), ctx.json(users))),
  rest.get('/api/recommendations', (req, res, ctx) => {
    const model = req.url.searchParams.get('model');
    const totalItems = model === 'vasp' ? 20 : 10;
    shuffle(movies);
    return res(ctx.status(200), ctx.json(movies.slice(0, totalItems)));
  }),
  rest.get('/api/interactions', (req, res, ctx) => {
    shuffle(movies);
    return res(ctx.status(200), ctx.json(movies.slice(0, randomInt(5, 30))));
  }),
];
