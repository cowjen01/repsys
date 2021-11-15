import { createStore, combineReducers } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import stateReconciler from 'redux-persist/lib/stateReconciler/autoMergeLevel2';

import studioReducer from './reducers/studio';
import recommendersReducer from './reducers/recommenders';
import settingsReducer from './reducers/settings';
import dialogsReducer from './reducers/dialogs';

const rootReducer = combineReducers({
  studio: studioReducer,
  recommenders: recommendersReducer,
  settings: settingsReducer,
  dialogs: dialogsReducer,
});

const persistConfig = {
  key: 'root',
  storage,
  stateReconciler,
  blacklist: ['dialogs'],
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

const store = createStore(persistedReducer);
const persistor = persistStore(store);

export default store;

export { persistor };
