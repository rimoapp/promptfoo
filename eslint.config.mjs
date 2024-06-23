// @ts-check

import eslint from '@eslint/js';
import importPlugin from "eslint-plugin-import";
import jest from 'eslint-plugin-jest';
import simpleImportSort from "eslint-plugin-simple-import-sort";
import unusedImports from 'eslint-plugin-unused-imports';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default [
  ...tseslint.config(eslint.configs.recommended, ...tseslint.configs.recommended),
  {
    ...jest.configs['flat/recommended'],
    rules: {
      ...jest.configs['flat/recommended'].rules,
      ...jest.configs['flat/style'].rules,
      'jest/consistent-test-it': 'error',
      'jest/expect-expect': 'error',
      'jest/prefer-expect-resolves': 'error',
      'jest/prefer-jest-mocked': 'error',
      'jest/require-to-throw-message': 'error',
    },
  },
  {
    ignores: [
      '**/src/web/nextui/_next/**/*',
      '**/src/web/nextui/.next/**/*',
      '**/src/web/nextui/out/**/*',
      'dist/**/*',
      'site/.docusaurus/**/*',
      'site/build/**/*',
    ],
  },
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: globals.node,
    },
    plugins: {
      'unused-imports': unusedImports,
      "simple-import-sort": simpleImportSort,
      "import": importPlugin,
    },
    rules: {
      '@typescript-eslint/ban-ts-comment': 0,
      '@typescript-eslint/ban-types': 0,
      '@typescript-eslint/no-explicit-any': 0,
      '@typescript-eslint/no-unused-vars': 0,
      '@typescript-eslint/no-var-requires': 0,
      'no-case-declarations': 0,
      'no-control-regex': 0,
      'no-empty': 0,
      'no-useless-escape': 0,
      'unused-imports/no-unused-imports': 'error',
      "import/first": "error",
      "import/newline-after-import": "error",
      "simple-import-sort/exports": "error",
      "simple-import-sort/imports": "error",
    },
  },
  {
    files: ['examples/**'],
    rules: {
      '@typescript-eslint/no-namespace': 0,
      '@typescript-eslint/no-var-requires': 0,
    },
  },
];
