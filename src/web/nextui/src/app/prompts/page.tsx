import React, { Suspense } from 'react';
import Prompts from './Prompts';

export default function Page() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div>
        <Prompts />
    </div>
    </Suspense>
  );
}
