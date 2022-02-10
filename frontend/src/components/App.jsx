import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { Routes, Route } from 'react-router-dom';

import Layout from './Layout';
import { ModelsEvaluation } from './models';
import { RecPreviews } from './recommenders';
import { DatasetEvaluation } from './dataset';
import { SettingsDialog } from './settings';
import Snackbar from './Snackbar';
import { fetchConfig } from '../reducers/config';
import { fetchUsers } from '../reducers/users';

function App() {
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(fetchConfig());
    dispatch(fetchUsers());
  }, []);

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<RecPreviews />} />
        <Route path="/models" element={<ModelsEvaluation />} />
        <Route path="/dataset" element={<DatasetEvaluation />} />
      </Routes>
      <Snackbar />
      <SettingsDialog />
    </Layout>
  );
}

export default App;
