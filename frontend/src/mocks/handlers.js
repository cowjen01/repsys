// eslint-disable-next-line import/no-extraneous-dependencies
import { rest } from 'msw';

export const handlers = [
  rest.get('/api/user', (req, res, ctx) =>
    res(
      ctx.status(200),
      ctx.json({
        username: 'admin',
      })
    )
  ),
];
