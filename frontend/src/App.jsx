import React from 'react';
import { Provider } from 'react-redux';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

import store from './store';
import Studio from './app/Studio';

function App() {
  return (
    <Provider store={store}>
      <DndProvider backend={HTML5Backend}>
        <Studio />
      </DndProvider>
    </Provider>
  );
}

export default App;
