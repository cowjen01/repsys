import React, { useMemo } from 'react';
import { Formik, Field } from 'formik';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Typography,
  LinearProgress,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import {
  addRecommender,
  recommendersSelector,
  updateRecommender,
} from '../../reducers/recommenders';
import { recEditDialogSelector, closeRecEditDialog, openSnackbar } from '../../reducers/dialogs';
import { TextField, SelectField, CheckboxField } from '../fields';
import { useGetModelsQuery } from '../../api';
import { capitalize } from '../../utils';
import ErrorAlert from '../ErrorAlert';

function RecEditDialog() {
  const dispatch = useDispatch();
  const dialog = useSelector(recEditDialogSelector);
  const recommenders = useSelector(recommendersSelector);

  const { index, open } = dialog;

  const models = useGetModelsQuery();

  const handleClose = () => {
    dispatch(closeRecEditDialog());
  };

  const handleSubmit = (values) => {
    const data = {
      ...values,
      itemsPerPage: parseInt(values.itemsPerPage, 10),
    };

    // the index can be 0, so !dialog.index is not enough
    if (index === null) {
      dispatch(addRecommender(data));
    } else {
      dispatch(
        updateRecommender({
          index,
          data,
        })
      );
    }

    dispatch(
      openSnackbar({
        message: 'All settings successfully applied!',
      })
    );

    handleClose();
  };

  const initialValues = useMemo(() => {
    if (!models.data) {
      return null;
    }

    if (index !== null) {
      const data = recommenders[index];

      // clear the model field if it does not exist anymore
      if (!models.data[data.model]) {
        data.model = '';
      }

      return data;
    }

    const defaultParams = Object.fromEntries(
      Object.entries(models.data).map(([modelKey, model]) => [
        modelKey,
        Object.fromEntries(
          Object.entries(model.params).map(([paramKey, param]) => [paramKey, param.default])
        ),
      ])
    );

    return {
      name: 'New Recommender',
      itemsPerPage: 4,
      itemsLimit: 20,
      model: Object.keys(models.data)[0],
      modelParams: defaultParams,
    };
  }, [index, models.data]);

  return (
    <Dialog open={open} transitionDuration={0} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Recommender settings</DialogTitle>
      <Formik
        initialValues={initialValues}
        enableReinitialize
        validate={(values) => {
          const errors = {};
          const requiredMessage = 'This field is required.';
          if (!values.name) {
            errors.name = requiredMessage;
          }
          if (index === null && recommenders.map(({ name }) => name).includes(values.name)) {
            errors.name = 'The name must be unique.';
          }
          if (!values.itemsLimit) {
            errors.itemsLimit = requiredMessage;
          }
          if (!values.model) {
            errors.model = requiredMessage;
          }
          return errors;
        }}
        onSubmit={(values, { setSubmitting }) => {
          handleSubmit(values);
          setSubmitting(false);
        }}
      >
        {({ submitForm, isSubmitting, values }) => {
          const modelData = values.model ? models.data[values.model] : null;
          return (
            <>
              <DialogContent>
                {models.isSuccess && (
                  <Grid container direction="column" spacing={2}>
                    <Grid item>
                      <Typography variant="subtitle2" component="div">
                        Appearance
                      </Typography>
                      <Field name="name" label="Name" fullWidth component={TextField} />
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Field
                            name="itemsPerPage"
                            label="Items per page"
                            component={SelectField}
                            options={[1, 3, 4]}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Field
                            name="itemsLimit"
                            label="Max number of items"
                            component={TextField}
                            type="number"
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                    <Grid item>
                      <Typography variant="subtitle2" component="div">
                        Model configuration
                      </Typography>
                      <Field
                        name="model"
                        label="Model"
                        component={SelectField}
                        options={Object.keys(models.data)}
                        displayEmpty
                      />
                      {modelData &&
                        modelData.params &&
                        Object.entries(modelData.params).map(([paramKey, param]) => {
                          const props = {
                            name: `modelParams.${values.model}.${paramKey}`,
                            label: capitalize(paramKey),
                          };
                          if (param.field === 'select') {
                            return (
                              <Field
                                key={paramKey}
                                component={SelectField}
                                options={param.options}
                                {...props}
                              />
                            );
                          }
                          if (param.field === 'checkbox') {
                            return <Field key={paramKey} component={CheckboxField} {...props} />;
                          }
                          return (
                            <Field
                              key={paramKey}
                              component={TextField}
                              type={param.field}
                              {...props}
                            />
                          );
                        })}
                    </Grid>
                  </Grid>
                )}
                {models.isLoading && <LinearProgress />}
                {models.isError && <ErrorAlert error={models.error} />}
              </DialogContent>
              <DialogActions>
                <Button onClick={handleClose}>Close</Button>
                <Button onClick={submitForm} disabled={isSubmitting} autoFocus>
                  Save
                </Button>
              </DialogActions>
            </>
          );
        }}
      </Formik>
    </Dialog>
  );
}

export default RecEditDialog;
