import React, { useMemo, useCallback } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
  Grid,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { Formik, Field } from 'formik';

import {
  darkModeSelector,
  itemFieldsSelector,
  setDarkMode,
  setItemFields,
} from '../../reducers/settings';
import { closeSettingsDialog, openSnackbar, settingsDialogSelector } from '../../reducers/dialogs';
import { configStatusSelector, datasetSelector } from '../../reducers/config';
import { SelectField, CheckboxField } from '../fields';

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function SettingsDialog() {
  const darkMode = useSelector(darkModeSelector);
  const dialogOpen = useSelector(settingsDialogSelector);
  const dispatch = useDispatch();
  const configStatus = useSelector(configStatusSelector);
  const dataset = useSelector(datasetSelector);
  const itemFields = useSelector(itemFieldsSelector);

  const handleClose = () => {
    dispatch(closeSettingsDialog());
  };

  const handleSubmit = useCallback(
    (values) => {
      dispatch(setDarkMode(values.darkMode));
      dispatch(setItemFields(values.itemFields));
      dispatch(
        openSnackbar({
          message: 'All settings applied!',
        })
      );
      handleClose();
    },
    [dispatch]
  );

  const itemColumnOptions = useMemo(() => {
    if (configStatus !== 'succeeded') {
      return [];
    }

    const options = dataset.columns.map((col) => ({ label: col, value: col }));

    return ['', ...options];
  }, [configStatus]);

  return (
    <Dialog open={dialogOpen} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Repsys Settings</DialogTitle>
      {configStatus === 'succeeded' ? (
        <Formik
          initialValues={{
            darkMode,
            itemFields,
          }}
          validate={(values) => {
            const errors = {};
            const requiredMessage = 'This field is required.';
            if (!values.itemFields.title) {
              errors['itemFields.title'] = requiredMessage;
            }
            return errors;
          }}
          onSubmit={(values, { setSubmitting }) => {
            handleSubmit(values);
            setSubmitting(false);
          }}
        >
          {({ submitForm, isSubmitting }) => (
            <>
              <DialogContent>
                <Grid container direction="column" spacing={2}>
                  <Grid item>
                    <Typography variant="subtitle2" component="div">
                      Recommendations
                    </Typography>
                    {['title', 'subtitle', 'caption', 'image', 'content'].map((field) => (
                      <Field
                        key={field}
                        name={`itemFields.${field}`}
                        label={`${capitalize(field)} field column`}
                        fullWidth
                        component={SelectField}
                        options={itemColumnOptions}
                      />
                    ))}
                  </Grid>
                  <Grid item>
                    <Typography variant="subtitle2" component="div">
                      Appearance
                    </Typography>
                    <Field name="darkMode" label="Nightshift mode" component={CheckboxField} />
                  </Grid>
                </Grid>
              </DialogContent>
              <DialogActions>
                <Button onClick={handleClose} color="secondary">
                  Close
                </Button>
                <Button onClick={submitForm} color="secondary" disabled={isSubmitting} autoFocus>
                  Save
                </Button>
              </DialogActions>
            </>
          )}
        </Formik>
      ) : (
        <DialogContent>Loading ...</DialogContent>
      )}
    </Dialog>
  );
}

export default SettingsDialog;
