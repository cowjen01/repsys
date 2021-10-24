import React, { useMemo, useState } from 'react';
// import pt from 'prop-types';
import Box from '@mui/material/Box';
import Chart from 'react-google-charts';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CheckIcon from '@mui/icons-material/Check';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';

import { fetchItems } from './api';

const colors = ['#2196f3', '#03a9f4', '#4caf50', '#9c27b0', '#ff9800', '#cddc39', '#ffeb3b'];

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function ModelMetrics({ onUserSelect, onInteractionsSelect }) {
  const { items: userEmbeddings, isLoading: isEmbeddingLoading } = fetchItems('/userSpace');
  const { items: itemsData, isLoading } = fetchItems('/items');

  const [selectedUser, setSelectedUser] = useState();
  const [interactions, setInteractions] = useState([]);
  const [activeTab, setActiveTab] = useState(0);

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const preparedData = useMemo(
    () =>
      userEmbeddings.map((p) => {
        let opacity = '0.5';
        if (!selectedUser || selectedUser === p.id) {
          opacity = '1';
        }
        return [p.x, p.y, `fill-color: ${colors[p.cluster]}; opacity: ${opacity}`, `User ${p.id}`];
      }),
    [userEmbeddings, selectedUser]
  );

  if (isLoading) {
    return null;
  }

  return (
    <Box
      sx={{
        minWidth: 450,
      }}
    >
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleChange} centered>
          <Tab label="Test User" />
          <Tab label="User Space" />
        </Tabs>
      </Box>
      <TabPanel value={activeTab} index={0}>
        <Typography variant="h6" component="div">
          Test User Simulator
        </Typography>
        <Typography variant="body2" component="div">
          Create a test user based on his interactions.
        </Typography>
        <Autocomplete
          multiple
          value={interactions}
          onChange={(event, newValue) => {
            setInteractions(newValue);
          }}
          openOnFocus
          isOptionEqualToValue={(option, value) => option.id === value.id}
          options={itemsData}
          getOptionLabel={(item) => item.title}
          sx={{ width: 400, marginBottom: 2, marginTop: 2 }}
          renderInput={(params) => (
            <TextField {...params} variant="filled" label="User interactions" />
          )}
        />
        <Button
          disabled={interactions.length === 0}
          color="secondary"
          startIcon={<CheckIcon />}
          variant="contained"
          onClick={() => onInteractionsSelect(interactions)}
        >
          Select Interactions
        </Button>
      </TabPanel>
      <TabPanel value={activeTab} index={1}>
        <Typography variant="h6" component="div">
          User Space Selector
        </Typography>
        <Typography variant="body2" component="div">
          Click on the point to select the user.
        </Typography>
        <Box
          sx={{
            marginTop: 2,
            marginBottom: 2,
          }}
        >
          <Chart
            width={400}
            height={400}
            chartType="ScatterChart"
            loader={<div>Loading Chart</div>}
            data={[
              ['X', 'Y', { type: 'string', role: 'style' }, { type: 'string', role: 'tooltip' }],
              ...preparedData,
            ]}
            options={{
              chartArea: {
                left: 10,
                top: 10,
                bottom: 10,
                right: 10,
                width: '100%',
                height: '100%',
              },
              dataOpacity: 1,
              baselineColor: 'transparent',
              enableInteractivity: true,
              pointSize: 8, // 3
              legend: 'none',
              hAxis: {
                textPosition: 'none',
                gridlines: {
                  color: 'transparent',
                },
              },
              vAxis: {
                textPosition: 'none',
                gridlines: {
                  color: 'transparent',
                },
              },
              // tooltip: {
              //   trigger: 'selection'
              // }
              // explorer: {
              //   keepInBounds: true,
              //   maxZoomIn: 3,
              //   maxZoomOut: 1,
              //   zoomDelta: 1.1
              // }
            }}
            chartEvents={[
              {
                eventName: 'select',
                callback: ({ chartWrapper }) => {
                  const chart = chartWrapper.getChart();
                  const selection = chart.getSelection();
                  if (selection.length === 1) {
                    const [selectedItem] = selection;
                    const { row } = selectedItem;
                    if (userEmbeddings[row]) {
                      setSelectedUser(userEmbeddings[row].id);
                    }
                  }
                },
              },
            ]}
          />
        </Box>
        <Button
          disabled={!selectedUser}
          color="secondary"
          startIcon={<CheckIcon />}
          variant="contained"
          onClick={() => onUserSelect(selectedUser)}
        >
          Select user {selectedUser}
        </Button>
      </TabPanel>
    </Box>
  );
}

export default ModelMetrics;
