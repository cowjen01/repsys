import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Tabs,
  Tab,
  Autocomplete,
  TextField,
  CircularProgress,
  Drawer,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import { useSelector, useDispatch } from 'react-redux';

import { favouriteUsersSelector, setCustomInteractions, setSelectedUser } from '../../reducers/app';
import { closeUserSelectDialog, userSelectDialogSelector } from '../../reducers/dialogs';
import TabPanel from '../TabPanel';
import { useGetUsersQuery, useGetItemsByTitleQuery } from '../../api';

let timerID;

function UserSelectDialog() {
  const dispatch = useDispatch();
  const dialogOpen = useSelector(userSelectDialogSelector);
  const favouriteUsers = useSelector(favouriteUsersSelector);

  const [currentUser, setCurrentUser] = useState(null);
  const [interactions, setInteractions] = useState([]);
  const [activeTab, setActiveTab] = useState(0);
  const [queryString, setQueryString] = useState('');

  const users = useGetUsersQuery({
    split: 'validation',
  });
  const items = useGetItemsByTitleQuery(queryString, {
    skip: queryString.length < 3,
  });

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setCurrentUser(null);
  };

  const handleDialogClose = () => {
    dispatch(closeUserSelectDialog());
    setCurrentUser(null);
    setActiveTab(0);
    setInteractions([]);
  };

  const handleUserSelect = () => {
    dispatch(setSelectedUser(currentUser));
    dispatch(setCustomInteractions([]));
    handleDialogClose();
  };

  const handleInteractionsSelect = () => {
    dispatch(setCustomInteractions(interactions));
    dispatch(setSelectedUser(null));
    handleDialogClose();
  };

  const handleQueryStringChange = (e, value) => {
    clearTimeout(timerID);
    timerID = setTimeout(() => {
      setQueryString(value);
    }, 300);
  };

  return (
    <Drawer anchor="right" open={dialogOpen} onClose={handleDialogClose}>
      <Box
        sx={{
          width: 450,
        }}
      >
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} centered>
            <Tab label="Users" />
            <Tab label="Simulator" />
            <Tab label="Favourites" />
          </Tabs>
        </Box>
        <TabPanel value={activeTab} index={0}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" component="div">
              User Selector
            </Typography>
            <Typography variant="body2" component="div">
              Select a user from the list of validation users.
            </Typography>
            <Autocomplete
              value={currentUser}
              loading={users.isLoading}
              onChange={(event, newValue) => setCurrentUser(newValue)}
              options={users.data || []}
              getOptionLabel={(user) => `User ${user}`}
              sx={{ marginBottom: 2, marginTop: 2 }}
              renderInput={(params) => (
                <TextField {...params} variant="filled" label="Selected user" />
              )}
            />
            <Button
              disabled={!currentUser}
              color="secondary"
              startIcon={<CheckIcon />}
              variant="contained"
              onClick={handleUserSelect}
            >
              Select user
            </Button>
          </Box>
        </TabPanel>
        <TabPanel value={activeTab} index={1}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" component="div">
              User Simulator
            </Typography>
            <Typography variant="body2" component="div">
              Create a test user based on his interactions.
            </Typography>
            <Autocomplete
              multiple
              value={interactions}
              onChange={(event, newValue) => setInteractions(newValue)}
              filterOptions={(x) => x}
              loading={items.isLoading}
              openOnFocus
              isOptionEqualToValue={(option, value) => option.id === value.id}
              options={items.data || []}
              getOptionLabel={(item) => item.title}
              sx={{ marginBottom: 2, marginTop: 2 }}
              onInputChange={handleQueryStringChange}
              renderInput={(params) => (
                <TextField
                  {...params}
                  variant="filled"
                  label="User interactions"
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {items.isFetching ? <CircularProgress color="inherit" size={20} /> : null}
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
              onClick={handleInteractionsSelect}
            >
              Select Interactions
            </Button>
          </Box>
        </TabPanel>
        <TabPanel value={activeTab} index={2}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" component="div">
              Favourite Users
            </Typography>
            <Typography variant="body2" component="div">
              Select a user from the list of favourites.
            </Typography>
            <Autocomplete
              value={currentUser}
              onChange={(event, newValue) => setCurrentUser(newValue)}
              options={favouriteUsers}
              getOptionLabel={(user) => `User ${user}`}
              sx={{ width: '100%', marginBottom: 2, marginTop: 2 }}
              renderInput={(params) => (
                <TextField {...params} variant="filled" label="Selected user" />
              )}
            />
            <Button
              disabled={!currentUser}
              color="secondary"
              startIcon={<CheckIcon />}
              variant="contained"
              onClick={handleUserSelect}
            >
              Select user
            </Button>
          </Box>
        </TabPanel>
      </Box>
    </Drawer>
  );
}

export default UserSelectDialog;
