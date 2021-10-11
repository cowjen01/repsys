// eslint-disable-next-line import/no-extraneous-dependencies
import { rest } from 'msw';

import { models, movies } from './data';

function shuffle(array) {
  return array.sort(() => Math.random() - 0.5);
}

export const handlers = [
  rest.get('/api/models', (req, res, ctx) => res(ctx.status(200), ctx.json(models))),
  rest.get('/api/recommendations', (req, res, ctx) => {
    const model = req.url.searchParams.get('model');
    const totalItems = model === 'vasp' ? 20 : 10;
    shuffle(movies);
    return res(ctx.status(200), ctx.json(movies.slice(0, totalItems)));
  }),
];
