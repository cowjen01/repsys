import React, { useMemo, useState, useEffect } from 'react';
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
import CircularProgress from '@mui/material/CircularProgress';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import { useSelector, useDispatch } from 'react-redux';

import { getRequest } from './api';
import { favouriteUsersSelector } from '../reducers/studio';

const colors = ['#2196f3', '#03a9f4', '#4caf50', '#9c27b0', '#ff9800', '#cddc39', '#ffeb3b'];

function TabPanel({ children, value, index, ...other }) {
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

function UserSearch({ onUserSelect, onInteractionsSelect, customInteractions }) {
  const [selectedUser, setSelectedUser] = useState(null);
  const [interactions, setInteractions] = useState(customInteractions);
  const [activeTab, setActiveTab] = useState(0);
  const [inputValue, setInputValue] = useState('');

  const favouriteUsers = useSelector(favouriteUsersSelector);

  const { items: userEmbeddings, isLoading: isEmbeddingLoading } = getRequest('/userSpace');
  const { items: itemsData, isLoading } = getRequest('/items', {
    query: inputValue,
  });

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const preparedData = useMemo(
    () =>
      userEmbeddings.map((p) => {
        let opacity = '0.5';
        if (!selectedUser || selectedUser.id === p.id) {
          opacity = '1';
        }
        return [
          p.x,
          p.y,
          `fill-color: ${colors[p.cluster]}; opacity: ${opacity}`,
          `User ${p.label}`,
        ];
      }),
    [userEmbeddings, selectedUser]
  );

  return (
    <Box
      sx={{
        minWidth: 450,
      }}
    >
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleChange} centered>
          <Tab label="Simulator" />
          <Tab label="Favourites" />
          <Tab label="Embeddings" />
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
          filterOptions={(x) => x}
          loading={isLoading}
          openOnFocus
          isOptionEqualToValue={(option, value) => option.id === value.id}
          options={itemsData}
          getOptionLabel={(item) => item.title}
          sx={{ width: 400, marginBottom: 2, marginTop: 2 }}
          onInputChange={(event, newInputValue) => {
            setInputValue(newInputValue);
          }}
          renderInput={(params) => (
            <TextField
              {...params}
              variant="filled"
              label="User interactions"
              InputProps={{
                ...params.InputProps,
                endAdornment: (
                  <>
                    {isLoading ? <CircularProgress color="inherit" size={20} /> : null}
                    {params.InputProps.endAdornment}
                  </>
                ),
              }}
            />
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
          Favourite Users
        </Typography>
        <Typography variant="body2" component="div">
          Select a user from the list of favourites.
        </Typography>
        <Autocomplete
          disablePortal
          value={selectedUser}
          onChange={(event, newValue) => {
            setSelectedUser(newValue);
          }}
          isOptionEqualToValue={(option, value) => option.id === value.id}
          options={favouriteUsers}
          getOptionLabel={(user) => `User ${user.label}`}
          sx={{ width: '100%', marginBottom: 2, marginTop: 2 }}
          renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
        />
        <Button
          disabled={!selectedUser}
          color="secondary"
          startIcon={<CheckIcon />}
          variant="contained"
          onClick={() => onUserSelect(selectedUser)}
        >
          Select user
        </Button>
      </TabPanel>
      <TabPanel value={activeTab} index={2}>
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
                      setSelectedUser({
                        id: userEmbeddings[row].id,
                        label: userEmbeddings[row].label,
                      });
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
          Select user {selectedUser ? selectedUser.label : ''}
        </Button>
      </TabPanel>
    </Box>
  );
}

export default UserSearch;
