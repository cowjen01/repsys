import { createStore, combineReducers, applyMiddleware } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import stateReconciler from 'redux-persist/lib/stateReconciler/autoMergeLevel2';
import thunkMiddleware from 'redux-thunk';

import rootReducer from './reducers/root';
import recommendersReducer from './reducers/recommenders';
import settingsReducer from './reducers/settings';
import dialogsReducer from './reducers/dialogs';
import interactionsReducer from './reducers/interactions';

const combinedReducers = combineReducers({
  root: rootReducer,
  recommenders: recommendersReducer,
  settings: settingsReducer,
  dialogs: dialogsReducer,
  interactions: interactionsReducer,
});

const persistConfig = {
  key: 'repsys',
  storage,
  stateReconciler,
  blacklist: ['dialogs', 'interactions'],
};

const persistedReducer = persistReducer(persistConfig, combinedReducers);

const store = createStore(persistedReducer, applyMiddleware(thunkMiddleware));
const persistor = persistStore(store);

export { store, persistor };
