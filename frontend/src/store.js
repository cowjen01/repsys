import { createStore, combineReducers } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import stateReconciler from 'redux-persist/lib/stateReconciler/autoMergeLevel2';

import rootReducer from './reducers/root';
import recommendersReducer from './reducers/recommenders';
import settingsReducer from './reducers/settings';
import dialogsReducer from './reducers/dialogs';

const combinedReducers = combineReducers({
  root: rootReducer,
  recommenders: recommendersReducer,
  settings: settingsReducer,
  dialogs: dialogsReducer,
});

const persistConfig = {
  key: 'repsys',
  storage,
  stateReconciler,
  blacklist: ['dialogs'],
};

const persistedReducer = persistReducer(persistConfig, combinedReducers);

const store = createStore(persistedReducer);
const persistor = persistStore(store);

export { store, persistor };
