import React, { useMemo } from 'react';
import pt from 'prop-types';
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
import { capitalize, mergeDeep } from '../../utils';
import ErrorAlert from '../ErrorAlert';

function ModelParams({ modelsData, model }) {
  const { params } = modelsData[model];

  if (!params) {
    return null;
  }

  return (
    <>
      {Object.entries(params).map(([key, { field, options }]) => {
        const props = {
          name: `modelParams.${model}.${key}`,
          label: capitalize(key),
        };
        if (field === 'select') {
          return (
            <Field key={key} displayEmpty component={SelectField} options={options} {...props} />
          );
        }
        if (field === 'checkbox') {
          return <Field key={key} component={CheckboxField} {...props} />;
        }
        return <Field key={key} component={TextField} type={field} {...props} />;
      })}
    </>
  );
}

ModelParams.propTypes = {
  modelsData: pt.any.isRequired,
  model: pt.string.isRequired,
};

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
      modelParams: Object.keys(models.data[values.model].params).reduce((obj, key) => {
        // eslint-disable-next-line no-param-reassign
        obj[key] = values.modelParams[values.model][key];
        return obj;
      }, {}),
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

    const defaultParams = Object.fromEntries(
      Object.entries(models.data).map(([modelKey, model]) => [
        modelKey,
        Object.fromEntries(
          Object.entries(model.params).map(([paramKey, param]) => [
            paramKey,
            param.default || (param.field !== 'checkbox' ? '' : false),
          ])
        ),
      ])
    );

    if (index !== null) {
      const data = recommenders[index];

      const mergedParams = mergeDeep(defaultParams, {
        [data.model]: data.modelParams,
      });

      const values = {
        ...data,
        modelParams: mergedParams,
      };

      // clear the model field if it does not exist anymore
      if (!models.data[data.model]) {
        values.model = '';
      }

      return values;
    }

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
        {({ submitForm, isSubmitting, values }) => (
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
                    {values.model && (
                      <ModelParams modelsData={models.data} model={values.model} />
                    )}
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
        )}
      </Formik>
    </Dialog>
  );
}

export default RecEditDialog;
