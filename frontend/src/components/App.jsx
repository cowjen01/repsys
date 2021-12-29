import React from 'react';

import { Routes, Route } from 'react-router-dom';
import Layout from './Layout';

import { ModelsEvaluation, RecPreviews } from './screens';
import { SettingsDialog } from './settings';
import Snackbar from './Snackbar';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<RecPreviews />} />
        <Route path="/models" element={<ModelsEvaluation />} />
      </Routes>
      <Snackbar />
      <SettingsDialog />
    </Layout>
  );
}

export default App;
