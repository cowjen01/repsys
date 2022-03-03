import React from 'react';
import ReactDOM from 'react-dom';

import '@fontsource/roboto/latin.css';

import Root from './Root';

if (process.env.NODE_ENV === 'development') {
  // eslint-disable-next-line global-require
  const { worker } = require('./mocks/browser');
  worker.start({ onUnhandledRequest: 'bypass' });
}

const app = React.createElement(Root);

ReactDOM.render(app, document.getElementById('root'));
