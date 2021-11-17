import { createStore, combineReducers, applyMiddleware } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import { composeWithDevTools } from 'redux-devtools-extension';
import storage from 'redux-persist/lib/storage';
import stateReconciler from 'redux-persist/lib/stateReconciler/autoMergeLevel2';
import thunkMiddleware from 'redux-thunk';

import rootReducer from './reducers/root';
import recommendersReducer from './reducers/recommenders';
import settingsReducer from './reducers/settings';
import dialogsReducer from './reducers/dialogs';
import interactionsReducer from './reducers/interactions';
import modelsReducer from './reducers/models';
import itemsReducer from './reducers/items';

const combinedReducers = combineReducers({
  root: rootReducer,
  recommenders: recommendersReducer,
  settings: settingsReducer,
  dialogs: dialogsReducer,
  interactions: interactionsReducer,
  models: modelsReducer,
  items: itemsReducer
});

const persistConfig = {
  key: 'repsys',
  storage,
  stateReconciler,
  blacklist: ['dialogs', 'interactions', 'models', 'items'],
};

const persistedReducer = persistReducer(persistConfig, combinedReducers);

const store = createStore(persistedReducer, composeWithDevTools(applyMiddleware(thunkMiddleware)));
const persistor = persistStore(store);

export { store, persistor };
